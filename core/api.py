import math
import os
import pickle
import time
import networkx as nx
import numpy as np
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.graph_model import add_objective_weights, path_metrics
from core.stop_selection import choose_spread_stops
from core.vrp import Customer, VRPInstance, capacity_feasible
from core.engine import solve_qpso, solve_ga_baseline, build_optimizer
from core.heuristics import solve_dynamic_heuristic
from core.traffic import apply_traffic_scenario
from core.ml_warmstart import predict_edge_probabilities

ox.settings.use_cache = True
ox.settings.cache_folder = "cache"

app = FastAPI(title="Quantum VRP Dispatch Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_CACHE = {}
PICKLE_CACHE_DIR = "cache/pickles"

class OptimizationRequest(BaseModel):
    place_name: str = Field("Salt Lake, Kolkata, India", min_length=1)
    customers_n: int = Field(10, ge=1, le=500)
    num_vehicles: int = Field(4, ge=1, le=100)
    vehicle_capacity: float = Field(25.0, gt=0)
    distance_weight: float = Field(0.2, ge=0)
    particles: int = Field(30, ge=3, le=1000)
    iterations: int = Field(50, ge=1, le=5000)
    traffic_mode: Literal["simulated", "live"] = "simulated"
    traffic_seed: int = 42
    sla_strictness_hours: float = Field(4.0, gt=0, le=24)

def prepare_osm_graph(place_name: str):
    if place_name in GRAPH_CACHE:
        return GRAPH_CACHE[place_name]

    os.makedirs(PICKLE_CACHE_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in place_name.lower())
    pickle_path = os.path.join(PICKLE_CACHE_DIR, f"{safe_name}.pkl")

    if os.path.exists(pickle_path):
        t_load = time.perf_counter()
        with open(pickle_path, "rb") as f:
            G = pickle.load(f)
        print(f"⏱️ Pickle load time: {time.perf_counter() - t_load:.4f}s")
    else:
        t_build = time.perf_counter()
        G_raw = ox.graph_from_place(place_name, network_type="drive", simplify=True)
        if G_raw.is_directed():
            scc = max(nx.strongly_connected_components(G_raw), key=len)
            G = G_raw.subgraph(scc).copy()
        else:
            cc = max(nx.connected_components(G_raw), key=len)
            G = G_raw.subgraph(cc).copy()

        for u, v, k, data in G.edges(keys=True, data=True):
            if 'distance_m' not in data:
                data['distance_m'] = float(data.get('length', 10.0))
            if 'travel_time_s' not in data:
                data['travel_time_s'] = data['distance_m'] / 8.33

        with open(pickle_path, "wb") as f:
            pickle.dump(G, f)
        print(f"⏱️ Built and dumped pickle cache in {time.perf_counter() - t_build:.4f}s")

    GRAPH_CACHE[place_name] = G
    return G

@app.post("/api/optimize")
def run_optimization(req: OptimizationRequest):
    t_total_start = time.perf_counter()

    try:
        G_base = prepare_osm_graph(req.place_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load map network: {str(e)}")

    historical_mode = 'rush_hour' if req.traffic_mode == 'live' else 'off_peak'
    G = apply_traffic_scenario(G_base, mode=historical_mode)

    try:
        depot_node, customer_nodes = choose_spread_stops(
            G, req.customers_n, seed=req.traffic_seed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rng = np.random.default_rng(req.traffic_seed)
    demands = rng.integers(1, 8, size=len(customer_nodes))
    total_demand = sum(demands)
    total_fleet_capacity = int(req.num_vehicles) * float(req.vehicle_capacity)

    if not capacity_feasible(demands, req.vehicle_capacity, req.num_vehicles):
        raise HTTPException(
            status_code=400,
            detail=f"Infeasible fleet: demand is {total_demand} and total capacity is {total_fleet_capacity}."
        )

    strictness_seconds = req.sla_strictness_hours * 3600.0
    start_of_day = 32400.0 
    
    customer_objs = []
    node_to_id = {depot_node: "Depot"}
    frontend_customers = []
    
    for i, (node, demand) in enumerate(zip(customer_nodes, demands)):
        window_duration = strictness_seconds
        offset = rng.uniform(0, 7200)
        c_ready_time = start_of_day + offset
        c_due_time = c_ready_time + window_duration
        
        customer_objs.append(Customer(
            node=node, demand=float(demand), ready_time=c_ready_time,
            due_time=c_due_time, service_time=300.0
        ))
        
        c_id = f"C{i+1}"
        node_to_id[node] = c_id
        frontend_customers.append({
            "id": c_id, "lat": G.nodes[node]['y'], "lng": G.nodes[node]['x'],
            "demand": float(demand), "ready_time": c_ready_time, "due_time": c_due_time
        })

    frontend_depot = {"id": "Depot", "lat": G.nodes[depot_node]['y'], "lng": G.nodes[depot_node]['x']}
    instance = VRPInstance(
        depot=depot_node, customers=customer_objs,
        vehicle_capacity=float(req.vehicle_capacity), num_vehicles=int(req.num_vehicles)
    )

    evaluate_shared, trace_path_shared, dist_mat_shared, time_mat_shared, costs_shared, dists_shared, tt_shared = build_optimizer(
        G, instance, time_weight=1.0, distance_weight=req.distance_weight
    )
    
    precomputed_eval = (evaluate_shared, trace_path_shared)
    precomputed_mats = (dist_mat_shared, time_mat_shared)
    precomputed_dicts = (costs_shared, dists_shared, tt_shared)

    # 🧠 STAGE 1: Neural Oracle & Constructive Baseline (Neural A*)
    try:
        edge_probs = predict_edge_probabilities(instance, G)
    except Exception as e:
        print(f"Warning: Neural Oracle prediction bypassed: {e}")
        edge_probs = None

    astar_result = solve_dynamic_heuristic(
        G, instance, method='astar', distance_weight=req.distance_weight,
        trace_path_func=trace_path_shared, precomputed_dicts=precomputed_dicts,
        edge_probs=edge_probs
    )

    # 🤝 STAGE 2: The Neural Hand-off
    node_to_idx = {c.node: i for i, c in enumerate(instance.customers)}
    base_permutation = []
    
    for route in astar_result.get("routes", []):
        for node in route:
            if node != instance.depot:
                base_permutation.append(node_to_idx[node])
                
    qpso_base_pos = np.zeros(len(instance.customers), dtype=float)
    denominator = max(1, len(instance.customers) - 1)
    for rank, cust_idx in enumerate(base_permutation):
        qpso_base_pos[cust_idx] = rank / denominator

    results = {'A*': astar_result}
    
    # 🏎️ STAGE 3: Metaheuristic Local Search (Improvement Phase)
    def run_ga():
        return 'GA', solve_ga_baseline(
            G, instance, particles=req.particles, iterations=req.iterations,
            seed=req.traffic_seed, distance_weight=req.distance_weight,
            precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats,
            initial_permutation=base_permutation
        )
        
    def run_qpso():
        return 'QPSO', solve_qpso(
            G, instance, particles=req.particles, iterations=req.iterations,
            seed=req.traffic_seed, distance_weight=req.distance_weight, use_ml_warmstart=False,
            precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats,
            initial_position=qpso_base_pos
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_ga), executor.submit(run_qpso)]
        for future in as_completed(futures):
            algo_name, res = future.result()
            results[algo_name] = res

    def first_nonfinite(value, path="result"):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)): return path
        if isinstance(value, dict):
            for key, child in value.items():
                if found := first_nonfinite(child, f"{path}.{key}"): return found
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if found := first_nonfinite(child, f"{path}[{index}]"): return found
        return None

    for algorithm, solution in results.items():
        if nonfinite_path := first_nonfinite(solution, algorithm):
            raise HTTPException(status_code=422, detail=f"{algorithm} produced non-finite routes ({nonfinite_path}).")

    expected_customer_nodes = {customer.node for customer in customer_objs}
    for algorithm, solution in results.items():
        served_nodes = [node for route in solution.get("routes", []) for node in route if node != depot_node]
        if set(served_nodes) != expected_customer_nodes or len(served_nodes) != len(expected_customer_nodes):
            raise HTTPException(status_code=422, detail=f"{algorithm} returned an incomplete customer set.")

    qpso_dist_km = results['QPSO'].get('distance_m', 0) / 1000.0
    astar_dist_km = results['A*'].get('distance_m', 0) / 1000.0
    dist_saved_km = astar_dist_km - qpso_dist_km
    
    liters_saved = dist_saved_km / 8.0
    rupees_saved = liters_saved * 100.0
    co2_saved_kg = liters_saved * 2.68

    palette = ["#D90429", "#0077B6", "#2A9D8F", "#F4A261", "#7209B7", "#FFB703"]
    cust_dict = {c.node: c for c in customer_objs}
    display_graph = add_objective_weights(G, time_weight=1.0, distance_weight=req.distance_weight)

    def serialize_solution(solution):
        routes_geojson = {"type": "FeatureCollection", "features": []}
        schedule_data = []
        for v_idx, route in enumerate(solution.get("routes", [])):
            color = palette[v_idx % len(palette)]
            coords = []
            stops_labels = [node_to_id.get(n, str(n)) for n in route]
            current_time = 32400.0
            route_start = current_time
            stops_timeline = []

            for a, b in zip(route[:-1], route[1:]):
                if a == b:
                    path, segment_time = [a], 0.0
                else:
                    path = solution.get("paths", {}).get((a, b))
                    if not path: path = [a, b]
                    segment_time, _ = path_metrics(display_graph, path, weight="_routing_weight")
                
                current_time += segment_time
                if b in cust_dict: current_time = max(current_time, cust_dict[b].ready_time)
                
                stops_timeline.append({
                    "stopId": node_to_id.get(b, str(b)),
                    "arrivalTime_s": round(current_time, 1),
                    "isDepot": b == instance.depot,
                })
                
                if b in cust_dict: current_time += cust_dict[b].service_time
                for node in path:
                    if node in G.nodes: coords.append([G.nodes[node]["x"], G.nodes[node]["y"]])

            if len(coords) > 1:
                routes_geojson["features"].append({
                    "type": "Feature",
                    "properties": {"vehicle": v_idx + 1, "color": color, "stops": stops_labels},
                    "geometry": {"type": "LineString", "coordinates": coords},
                })
            schedule_data.append({
                "vehicle": v_idx + 1, "color": color, "stops": stops_labels,
                "timeline": stops_timeline, "totalTime_min": round((current_time - route_start) / 60.0, 1),
            })
        return routes_geojson, schedule_data

    routes_by_algorithm = {}
    schedules_by_algorithm = {}
    for algorithm, solution in results.items():
        routes_by_algorithm[algorithm], schedules_by_algorithm[algorithm] = serialize_solution(solution)

    print(f"🚀 TOTAL ENDPOINT TIME: {time.perf_counter() - t_total_start:.4f}s")

    return {
        "locations": {"depot": frontend_depot, "customers": frontend_customers},
        "business_impact": {
            "distance_saved_km": round(dist_saved_km, 2), "rupees_saved": round(rupees_saved, 0),
            "liters_saved": round(liters_saved, 1), "co2_saved_kg": round(co2_saved_kg, 1),
            "qpso_dist_km": round(qpso_dist_km, 2), "astar_dist_km": round(astar_dist_km, 2)
        },
        "algorithms": {
            algorithm: {
                "score": round(solution["score"], 2), "distance_km": round(solution["distance_m"] / 1000.0, 2),
                "travel_time_min": round(solution["travel_time_s"] / 60.0, 1), "total_duration_min": round(solution["total_duration_s"] / 60.0, 1),
                "tw_penalty": round(solution.get("tw_penalty", 0.0), 2), "total_lateness_s": round(solution.get("total_lateness_s", 0.0), 1),
                "history": solution["history"],
            }
            for algorithm, solution in results.items()
        },
        "routes": routes_by_algorithm["QPSO"], "schedules": schedules_by_algorithm["QPSO"],
        "routes_by_algorithm": routes_by_algorithm, "schedules_by_algorithm": schedules_by_algorithm,
    }