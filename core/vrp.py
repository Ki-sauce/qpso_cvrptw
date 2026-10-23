from dataclasses import dataclass
from typing import List, Dict, Any
import math
import numpy as np
from numba import jit, prange

@dataclass
class Customer:
    node: int
    demand: float
    ready_time: float = 0.0
    due_time: float = 86400.0
    service_time: float = 300.0

@dataclass
class VRPInstance:
    depot: int
    customers: List[Customer]
    vehicle_capacity: float
    num_vehicles: int
    time_limit_s: float | None = None

def capacity_feasible(demands, vehicle_capacity, num_vehicles):
    if vehicle_capacity <= 0 or num_vehicles < 1:
        return False
    values = sorted((float(demand) for demand in demands), reverse=True)
    if any(value < 0 or value > vehicle_capacity for value in values):
        return False
    if len(values) > 24:
        bins = [0.0] * int(num_vehicles)
        for demand in values:
            for index, load in enumerate(bins):
                if load + demand <= vehicle_capacity:
                    bins[index] += demand
                    break
            else:
                return False
        return True
    bins = [0.0] * int(num_vehicles)
    def place(index):
        if index == len(values):
            return True
        demand = values[index]
        seen = set()
        for bin_index, load in enumerate(bins):
            if load in seen or load + demand > vehicle_capacity:
                continue
            seen.add(load)
            bins[bin_index] += demand
            if place(index + 1):
                return True
            bins[bin_index] -= demand
        return False
    return place(0)

def construct_feasible_order(instance: VRPInstance):
    demands = [customer.demand for customer in instance.customers]
    if not capacity_feasible(demands, instance.vehicle_capacity, instance.num_vehicles):
        return None
    if len(demands) <= 24:
        bins = [([], 0.0) for _ in range(instance.num_vehicles)]
        ranked = sorted(enumerate(demands), key=lambda item: (-float(item[1]), item[0]))
        def place(position):
            if position == len(ranked):
                return True
            index, demand = ranked[position]
            seen_loads = set()
            for bin_index, (members, load) in enumerate(bins):
                if load in seen_loads or load + demand > instance.vehicle_capacity:
                    continue
                seen_loads.add(load)
                bins[bin_index] = (members + [index], load + demand)
                if place(position + 1):
                    return True
                bins[bin_index] = (members, load)
            return False
        if not place(0):
            return None
        return [index for members, _ in bins for index in members]
    bins = [(0.0, []) for _ in range(instance.num_vehicles)]
    ranked = sorted(enumerate(instance.customers), key=lambda item: (-float(item[1].demand), item[0]))
    for index, customer in ranked:
        target = min(
            ((bin_index, load) for bin_index, (load, _) in enumerate(bins) if load + customer.demand <= instance.vehicle_capacity),
            key=lambda item: item[1],
            default=None,
        )
        if target is None:
            return None
        bin_index = target[0]
        load, members = bins[bin_index]
        bins[bin_index] = (load + customer.demand, members + [index])
    return [index for _, members in bins for index in members]

def split_random_key_solution(order, instance: VRPInstance):
    if instance.vehicle_capacity <= 0 or not order:
        return None
    
    routes = []
    current = [instance.depot]
    load = 0.0
    
    for customer in order:
        # STRICT BOUND: Never allow load to exceed vehicle_capacity during splitting
        if load + customer.demand <= instance.vehicle_capacity:
            current.append(customer.node)
            load += customer.demand
        else:
            current.append(instance.depot)
            routes.append(current)
            current = [instance.depot, customer.node]
            load = customer.demand

    current.append(instance.depot)
    routes.append(current)

    # Pad with empty routes up to num_vehicles
    while len(routes) < instance.num_vehicles:
        routes.append([instance.depot, instance.depot])

    return routes

@jit(nopython=True, fastmath=True, nogil=True)
def evaluate_routes_kernel(route_indices_flat, route_lens, demands, tw_starts, tw_ends, service_times,
                           dist_matrix, time_matrix, vehicle_capacity, time_weight, distance_weight,
                           dispatch_start_s, lateness_penalty_per_min):
    total_time = 0.0
    total_distance = 0.0
    total_service = 0.0
    total_waiting = 0.0
    total_lateness = 0.0
    total_capacity_penalty = 0.0
    offset = 0
    
    for r_idx in range(len(route_lens)):
        length = route_lens[r_idx]
        if length <= 2:
            offset += length
            continue
        
        route = route_indices_flat[offset:offset+length]
        offset += length
        
        current_time = dispatch_start_s
        route_demand = 0.0
        
        for k in range(len(route) - 1):
            u_idx = route[k]
            v_idx = route[k+1]
            
            if u_idx == v_idx:
                continue
                
            travel_time = time_matrix[u_idx, v_idx]
            dist = dist_matrix[u_idx, v_idx]
            
            if not np.isfinite(travel_time) or not np.isfinite(dist):
                return 1e12, 1e12, 1e12, 1e12, 1e12, 1e12, 1e12
            total_time += travel_time
            total_distance += dist
            current_time += travel_time
            
            if v_idx > 0:
                cust_real_idx = v_idx - 1
                route_demand += demands[cust_real_idx]
                r_ready = tw_starts[cust_real_idx]
                r_due = tw_ends[cust_real_idx]
                r_serv = service_times[cust_real_idx]
                
                if current_time < r_ready:
                    total_waiting += (r_ready - current_time)
                    current_time = r_ready
                if current_time > r_due:
                    total_lateness += (current_time - r_due)
                    
                current_time += r_serv
                total_service += r_serv
                
        # SOFT PENALTY GRADIENT: Penalize capacity overages gradually instead of returning 1e18
        if route_demand > vehicle_capacity:
            total_capacity_penalty += (route_demand - vehicle_capacity) * 100000.0

    tw_penalty = (total_lateness / 60.0) * lateness_penalty_per_min
    objective = (
        time_weight * (total_time + total_waiting + total_service) +
        distance_weight * total_distance +
        tw_penalty +
        total_capacity_penalty
    )
    return objective, total_time, total_distance, total_service, total_waiting, tw_penalty, total_lateness

def evaluate_routes(routes, instance, costs, distances, paths,
                     time_weight=1.0, distance_weight=0.0,
                    congestion_weight=0.0, travel_times=None,
                    dispatch_start_s=32400.0,
                    lateness_penalty_per_min=10000.0):
    if routes is None:
        return math.inf, {"objective": math.inf}
    travel_times = travel_times or costs
    total_time = 0.0
    total_distance = 0.0
    total_service = 0.0
    total_waiting = 0.0
    total_lateness = 0.0
    customer_map = {c.node: c for c in instance.customers}
    for route in routes:
        current_time = dispatch_start_s
        for u, v in zip(route[:-1], route[1:]):
            c = costs.get((u, v), math.inf)
            travel_time = travel_times.get((u, v), math.inf)
            d = distances.get((u, v), math.inf)
            if not all(math.isfinite(value) for value in (c, travel_time, d)):
                return math.inf, {"objective": math.inf}
            total_time += travel_time
            total_distance += d
            current_time += travel_time
            customer = customer_map.get(v)
            if customer is not None:
                if current_time < customer.ready_time:
                    total_waiting += customer.ready_time - current_time
                    current_time = customer.ready_time
                if current_time > customer.due_time:
                    total_lateness += current_time - customer.due_time
                current_time += customer.service_time
                total_service += customer.service_time
    tw_penalty = (total_lateness / 60.0) * lateness_penalty_per_min
    objective = (
        time_weight * (total_time + total_waiting + total_service) +
        distance_weight * total_distance +
        congestion_weight * 0.0 +
        tw_penalty
    )
    return objective, {
        "travel_time_s": total_time,
        "distance_m": total_distance,
        "service_time_s": total_service,
        "waiting_time_s": total_waiting,
        "total_duration_s": total_time + total_waiting + total_service,
        "tw_penalty": tw_penalty,
        "total_lateness_s": total_lateness,
        "objective": objective,
    }

def optimize_route_2opt(route, costs):
    if len(route) < 4:
        return route
    best_route = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1: continue 
                new_route = best_route[:i] + list(reversed(best_route[i:j + 1])) + best_route[j + 1:]
                old_cost = sum(costs.get((best_route[k], best_route[k+1]), math.inf) for k in range(len(best_route)-1))
                new_cost = sum(costs.get((new_route[k], new_route[k+1]), math.inf) for k in range(len(new_route)-1))
                if new_cost < old_cost:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route