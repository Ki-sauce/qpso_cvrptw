import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def _evaluate_single_particle_jit(
    particle, demands, capacity, num_vehicles,
    distance_matrix, time_matrix, ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    """
    Decodes the random key into a sequence and evaluates fitness in a single O(N) pass.
    Node 0 is strictly the depot. Customers are indexed 1 to N.
    """
    n_customers = len(particle)
    # Argsort yields customer indices (0 to N-1). Add 1 to offset for the depot (Node 0).
    order = np.argsort(particle) + 1 
    
    total_time = 0.0
    total_distance = 0.0
    total_service = 0.0
    total_waiting = 0.0
    total_lateness = 0.0
    
    current_node = 0
    current_load = 0.0
    current_time = dispatch_start_s
    vehicles_used = 1
    
    for i in range(n_customers):
        cust = order[i]
        demand = demands[cust]
        
        # Check Capacity for this vehicle
        if current_load + demand <= capacity:
            travel_t = time_matrix[current_node, cust]
            travel_d = distance_matrix[current_node, cust]
            current_time += travel_t
            current_load += demand
        else:
            # Vehicle full: Return to depot, dispatch new vehicle
            vehicles_used += 1
            if vehicles_used > num_vehicles:
                return np.inf # Hard constraint violation
                
            # Close old route
            total_time += time_matrix[current_node, 0]
            total_distance += distance_matrix[current_node, 0]
            
            # Start new route
            travel_t = time_matrix[0, cust]
            travel_d = distance_matrix[0, cust]
            current_time = dispatch_start_s + travel_t
            current_load = demand
            
        total_time += travel_t
        total_distance += travel_d
        
        # Enforce Time Windows
        if current_time < ready_times[cust]:
            total_waiting += (ready_times[cust] - current_time)
            current_time = ready_times[cust]
        elif current_time > due_times[cust]:
            total_lateness += (current_time - due_times[cust])
            
        # Unload
        current_time += service_times[cust]
        total_service += service_times[cust]
        current_node = cust
        
    # Final return to depot for the last active vehicle
    total_time += time_matrix[current_node, 0]
    total_distance += distance_matrix[current_node, 0]
    
    tw_penalty = (total_lateness / 60.0) * lateness_penalty
    objective = (time_weight * (total_time + total_waiting + total_service) + 
                 distance_weight * total_distance + 
                 tw_penalty)
                 
    return objective

@njit(parallel=True, fastmath=True, nogil=True)
def evaluate_swarm_jit(
    particles, demands, capacity, num_vehicles,
    distance_matrix, time_matrix, ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    """
    Releases the GIL and evaluates the entire QPSO swarm concurrently across all physical cores.
    """
    n_particles = particles.shape[0]
    scores = np.empty(n_particles, dtype=np.float32)
    
    # prange distributes particles across hardware threads
    for i in prange(n_particles):
        scores[i] = _evaluate_single_particle_jit(
            particles[i], demands, capacity, num_vehicles,
            distance_matrix, time_matrix, ready_times, due_times, service_times,
            time_weight, distance_weight, lateness_penalty, dispatch_start_s
        )
    return scores