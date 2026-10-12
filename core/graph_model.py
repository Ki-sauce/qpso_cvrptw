import os
import networkx as nx
import numpy as np
import osmnx as ox
import math
from scipy.sparse.csgraph import dijkstra

CACHE_DIR = "cache/graphs"

def load_osm_network(place):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in place.lower())
    file_path = os.path.join(CACHE_DIR, f"{safe_name}.graphml")
    
    if os.path.exists(file_path):
        G = ox.load_graphml(file_path)
    else:
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        G = ox.project_graph(G)
        ox.save_graphml(G, file_path)

    for _, _, data in G.edges(data=True):
        # Force string values loaded from GraphML back into floats
        dist = float(data.get("distance_m", data.get("length", 10.0)))
        data["distance_m"] = dist
        
        if "travel_time_s" in data:
            data["travel_time_s"] = float(data["travel_time_s"])
        else:
            data["travel_time_s"] = dist / 8.33
            
    return G

def nearest_node(G, x, y): return ox.distance.nearest_nodes(G, X=x, Y=y)
def edge_weighted_subgraph(G, nodes): return G.subgraph(nodes).copy()

def edge_attributes(G, u, v, weight="travel_time_s"):
    data = G.get_edge_data(u, v)
    if not data: return None
    if G.is_multigraph():
        return min(data.values(), key=lambda attrs: float(attrs.get(weight, float("inf"))))
    return data

def path_metrics(G, path, weight="travel_time_s"):
    total_time, total_distance = 0.0, 0.0
    for u, v in zip(path[:-1], path[1:]):
        attrs = edge_attributes(G, u, v, weight=weight)
        if attrs is None: return float("inf"), float("inf")
        total_time += float(attrs.get("travel_time_s", 0.0))
        total_distance += float(attrs.get("distance_m", attrs.get("length", 0.0)))
    return total_time, total_distance

def add_objective_weights(G, time_weight=1.0, distance_weight=0.0):
    weighted = G.copy()
    for _, _, data in weighted.edges(data=True):
        travel_time = float(data.get("travel_time_s", float("inf")))
        distance = float(data.get("distance_m", data.get("length", 0.0)))
        data["_routing_weight"] = time_weight * travel_time + distance_weight * distance
    return weighted

def build_all_matrices_scipy(G, instance, weight="_routing_weight"):
    stops = [instance.depot] + [c.node for c in instance.customers]
    n_stops = len(stops)
    
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = np.array(nodes)
    stop_indices = np.array([node_to_idx[n] for n in stops])

    try:
        adj_weight = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=weight)
        adj_time = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="travel_time_s")
        adj_dist = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="distance_m")
    except AttributeError:
        adj_weight = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight=weight)
        adj_time = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight="travel_time_s")
        adj_dist = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight="distance_m")

    # 1. Compute raw distances to all graph nodes
    cost_mat_raw, predecessors = dijkstra(csgraph=adj_weight, directed=True, indices=stop_indices, return_predecessors=True)
    time_mat_raw = dijkstra(csgraph=adj_time, directed=True, indices=stop_indices)
    dist_mat_raw = dijkstra(csgraph=adj_dist, directed=True, indices=stop_indices)

    # 2. Slice strictly to (N_stops x N_stops) sub-matrices
    cost_mat = np.ascontiguousarray(cost_mat_raw[:, stop_indices], dtype=np.float64)
    time_mat = np.ascontiguousarray(time_mat_raw[:, stop_indices], dtype=np.float64)
    dist_mat = np.ascontiguousarray(dist_mat_raw[:, stop_indices], dtype=np.float64)

    # 3. Populate fast lookup dictionaries
    costs, travel_times, distances = {}, {}, {}
    for i, u in enumerate(stops):
        for j, v in enumerate(stops):
            costs[(u, v)] = float(cost_mat[i, j])
            travel_times[(u, v)] = float(time_mat[i, j])
            distances[(u, v)] = float(dist_mat[i, j])

    # 4. Lazy physical coordinate trace function
    def trace_path(u, v):
        if u == v: return [u]
        j = node_to_idx.get(v)
        if j is None: return []
        try: stop_i = stops.index(u)
        except ValueError: return []
        
        path_idx = []
        curr = j
        source_idx = stop_indices[stop_i]
        while curr != source_idx and curr >= 0:
            path_idx.append(curr)
            curr = predecessors[stop_i, curr]
        if curr < 0: return []
        path_idx.append(source_idx)
        path_idx.reverse()
        return idx_to_node[path_idx].tolist()

    return costs, distances, travel_times, dist_mat, time_mat, trace_path