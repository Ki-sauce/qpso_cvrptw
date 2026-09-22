from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass
class Customer:
    node: int
    demand: float
    # Add these three lines for Time Windows:
    ready_time: float = 0.0
    due_time: float = 86400.0  # Default to end of the day (24 hours in seconds)
    service_time: float = 300.0 # 5 minutes to unload at the location


@dataclass
class VRPInstance:
    depot: int
    customers: List[Customer]
    vehicle_capacity: float
    num_vehicles: int
    time_limit_s: float | None = None


def capacity_feasible(demands, vehicle_capacity, num_vehicles):
    """Exact small bin-packing feasibility check for the requested fleet."""
    if vehicle_capacity <= 0 or num_vehicles < 1:
        return False
    values = sorted((float(demand) for demand in demands), reverse=True)
    if any(value < 0 or value > vehicle_capacity for value in values):
        return False

    # Exact bin packing is exponential. Use exact search for small instances
    # and a deterministic first-fit-decreasing feasibility check for large
    # interactive requests so validation never hangs the API.
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
    """Construct a customer order whose split decoder uses the fleet."""
    demands = [customer.demand for customer in instance.customers]
    if not capacity_feasible(
        demands,
        instance.vehicle_capacity,
        instance.num_vehicles,
    ):
        return None

    # For small requests, produce the actual packing assignment rather than
    # relying on first-fit decreasing.  FFD is fast, but it can reject a
    # packing that is feasible with the requested number of vehicles.  The
    # same bounded exact search used by capacity_feasible is safe here.
    if len(demands) <= 24:
        bins = [([], 0.0) for _ in range(instance.num_vehicles)]
        ranked = sorted(
            enumerate(demands),
            key=lambda item: (-float(item[1]), item[0]),
        )

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

    # Large interactive requests use the bounded deterministic approximation.
    bins = [(0.0, []) for _ in range(instance.num_vehicles)]
    ranked = sorted(
        enumerate(instance.customers),
        key=lambda item: (-float(item[1].demand), item[0]),
    )
    for index, customer in ranked:
        target = min(
            (
                (bin_index, load)
                for bin_index, (load, _) in enumerate(bins)
                if load + customer.demand <= instance.vehicle_capacity
            ),
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
    """
    Decode a customer permutation into <= num_vehicles capacity-feasible routes.
    Each route starts and ends at the depot.
    """
    if instance.vehicle_capacity <= 0:
        return None
    if any(c.demand < 0 or c.demand > instance.vehicle_capacity for c in order):
        return None

    routes = []
    current = [instance.depot]
    load = 0.0

    for customer in order:
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

    if len(routes) > instance.num_vehicles:
        return None

    while len(routes) < instance.num_vehicles:
        routes.append([instance.depot, instance.depot])

    return routes


def evaluate_routes(routes, instance, costs, distances, paths, 
                    time_weight=1.0, distance_weight=0.0,
                    congestion_weight=0.0, travel_times=None,
                    dispatch_start_s=32400.0,
                    lateness_penalty_per_min=10000.0):
    """
    Objective = weighted travel time + distance.
    Congestion is already reflected in travel_time_s; congestion_weight is
    retained as a hook for a separate congestion penalty.
    """
    if routes is None:
        return math.inf, {"objective": math.inf}

    travel_times = travel_times or costs

    total_time = 0.0
    total_distance = 0.0
    total_service = 0.0
    total_waiting = 0.0
    total_lateness = 0.0

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
            customer = next((item for item in instance.customers if item.node == v), None)
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

# Append this function to the bottom of core/vrp.py

def optimize_route_2opt(route, costs):
    """
    Applies 2-opt local search to a single vehicle route to eliminate 
    overlapping edges and minimize local transit cost.
    """
    if len(route) < 4:
        return route
        
    best_route = list(route)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            # Keep the depot fixed at both ends of the route.
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1: continue 
                
                # Reverse the complete segment between i and j.
                new_route = best_route[:i] + list(reversed(best_route[i:j + 1])) + best_route[j + 1:]
                
                # Quick cost evaluation of the sub-segments
                old_cost = sum(costs.get((best_route[k], best_route[k+1]), math.inf) for k in range(len(best_route)-1))
                new_cost = sum(costs.get((new_route[k], new_route[k+1]), math.inf) for k in range(len(new_route)-1))
                
                if new_cost < old_cost:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
                
    return best_route
