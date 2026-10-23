import numpy as np
import os
from core.vrp import evaluate_routes_kernel

def _evaluate_particle(
    particle_position, dist_matrix, time_matrix, 
    demands, tw_starts, tw_ends, service_times, 
    v_cap, n_veh, d_idx, 
    t_wt, d_wt, disp_s, lat_pen
):
    """Evaluates a single particle's 1D position vector using argsort decoding."""
    sorted_order_indices = np.argsort(particle_position)
    
    route_lens_list = []
    route_flat_list = [int(d_idx)]
    curr_load = 0.0
    current_route_len = 1
    
    for cust_idx in sorted_order_indices:
        c_idx = int(cust_idx)
        c_demand = float(demands[c_idx])
        matrix_node_idx = c_idx + 1  # 0 is depot, 1..N are customers
        
        if curr_load + c_demand <= float(v_cap):
            route_flat_list.append(matrix_node_idx)
            curr_load += c_demand
            current_route_len += 1
        else:
            route_flat_list.append(int(d_idx))
            current_route_len += 1
            route_lens_list.append(current_route_len)
            route_flat_list.append(int(d_idx))
            route_flat_list.append(matrix_node_idx)
            curr_load = c_demand
            current_route_len = 3
            
    route_flat_list.append(int(d_idx))
    current_route_len += 1
    route_lens_list.append(current_route_len)
    
    while len(route_lens_list) < int(n_veh):
        route_flat_list.extend([int(d_idx), int(d_idx)])
        route_lens_list.append(2)
        
    route_flat_arr = np.array(route_flat_list, dtype=np.int64)
    route_lens_arr = np.array(route_lens_list, dtype=np.int64)
    
    obj, _, _, _, _, _, _ = evaluate_routes_kernel(
        route_flat_arr, route_lens_arr, demands, tw_starts, tw_ends, service_times,
        dist_matrix, time_matrix, float(v_cap), float(t_wt), float(d_wt),
        float(disp_s), float(lat_pen)
    )
    return float(obj)


class QPSO:
    def __init__(self, evaluate=None, n_particles=40, iterations=100, seed=42, max_workers=None,
                 dist_matrix=None, time_matrix=None, demands=None,
                 tw_starts=None, tw_ends=None, service_times=None, vehicle_capacity=None,
                 num_vehicles=None, depot_idx=0, time_weight=1.0, distance_weight=0.0):
        self.evaluate = evaluate
        self.n_particles = n_particles
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        
        self.dist_matrix = dist_matrix
        self.time_matrix = time_matrix
        self.demands = demands
        self.tw_starts = tw_starts
        self.tw_ends = tw_ends
        self.service_times = service_times
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.depot_idx = depot_idx
        self.time_weight = time_weight
        self.distance_weight = distance_weight
        
    def optimize(self, dimensions, initial_position=None, initial_population=None):
        M = self.n_particles
        D = dimensions
        
        # Initialize swarm positions spread cleanly in [0, 1]
        particles = self.rng.uniform(0.0, 1.0, (M, D))
        
        if initial_population is not None:
            init_pop = np.asarray(initial_population, dtype=float)
            n_init = min(len(init_pop), M)
            particles[:n_init] = np.clip(init_pop[:n_init], 0.0, 1.0)
        elif initial_position is not None:
            initial_position = np.asarray(initial_position, dtype=float)
            particles[0] = np.clip(initial_position, 0.0, 1.0)
            
        pbest = particles.copy()
        pbest_scores = np.full(M, np.inf)
        
        eval_args = (
            self.dist_matrix, self.time_matrix, 
            self.demands, self.tw_starts, self.tw_ends, self.service_times, 
            float(self.vehicle_capacity), int(self.num_vehicles), int(self.depot_idx),
            float(self.time_weight), float(self.distance_weight), 32400.0, 10000.0
        )
        
        # Sequential inner evaluation loop (eliminates GIL multi-threading bottleneck)
        for i in range(M):
            pbest_scores[i] = _evaluate_particle(particles[i], *eval_args)
            
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        history = [gbest_score]
        
        for it in range(self.iterations):
            progress = it / max(1, self.iterations - 1)
            # Contraction-Expansion coefficient decay
            alpha = 1.0 - 0.7 * progress
            
            # Elite Mean Best calculation
            sorted_indices = np.argsort(pbest_scores)
            n_elite = max(5, int(0.3 * M))
            elite_indices = sorted_indices[:n_elite]
            elite_pbest = pbest[elite_indices]
            elite_scores = pbest_scores[elite_indices]
            
            ptp = np.ptp(elite_scores)
            denom = ptp if ptp > 1e-8 else 1.0
            score_diffs = elite_scores - elite_scores[0]
            exp_scores = np.exp(-score_diffs / denom)
            weights = exp_scores / np.sum(exp_scores)
            mbest = np.sum(elite_pbest * weights[:, np.newaxis], axis=0)
            
            for i in range(M):
                phi = self.rng.uniform(0.0, 1.0, D)
                p = phi * pbest[i] + (1.0 - phi) * gbest
                
                u = self.rng.uniform(1e-10, 1.0 - 1e-10, D)
                sign = np.where(self.rng.random(D) > 0.5, 1.0, -1.0)
                
                # Quantum position update step
                particles[i] = p + sign * alpha * np.abs(mbest - particles[i]) * np.log(1.0 / u)
                
                # FIX 1: Rank Re-normalization to prevent [0, 1] clipping flatline collapse
                ranks = np.argsort(np.argsort(particles[i]))
                particles[i] = ranks / float(max(1, D - 1))
                
                score = _evaluate_particle(particles[i], *eval_args)
                
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score
                    if score < gbest_score:
                        gbest = particles[i].copy()
                        gbest_score = score

            # FIX 2: Periodic 2-Opt local permutation swap on Global Best
            if (it + 1) % 5 == 0 and D >= 4:
                perm = np.argsort(gbest)
                i_swap, j_swap = sorted(self.rng.choice(D, size=2, replace=False))
                perm[i_swap], perm[j_swap] = perm[j_swap], perm[i_swap]
                
                cand_pos = np.empty(D, dtype=np.float64)
                cand_pos[perm] = np.linspace(0.0, 1.0, D)
                
                cand_score = _evaluate_particle(cand_pos, *eval_args)
                if cand_score < gbest_score:
                    gbest = cand_pos.copy()
                    gbest_score = cand_score

            history.append(gbest_score)
            
        return gbest, gbest_score, history