import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from core.vrp import evaluate_routes_kernel

def _thread_evaluate(args):
    """Fast evaluation of direct customer permutations."""
    (
        particle_position, dist_matrix, time_matrix, 
        demands, tw_starts, tw_ends, service_times, 
        v_cap, n_veh, d_idx, 
        t_wt, d_wt, disp_s, lat_pen
    ) = args

    sorted_order_indices = np.argsort(particle_position)

    route_lens_list = []
    route_flat_list = [int(d_idx)]
    curr_load = 0.0
    current_route_len = 1

    for cust_idx in sorted_order_indices:
        c_idx = int(cust_idx)
        c_demand = float(demands[c_idx])
        matrix_node_idx = c_idx + 1

        if curr_load + c_demand <= float(v_cap):
            route_flat_list.append(matrix_node_idx)
            curr_load += c_demand
            current_route_len += 1
        else:
            route_flat_list.append(int(d_idx))
            current_route_len += 1
            route_lens_list.append(current_route_len)
            
            # Start new route with depot and first customer (exactly 2 nodes)
            route_flat_list.append(int(d_idx))
            route_flat_list.append(matrix_node_idx)
            curr_load = c_demand
            current_route_len = 2  # ⚡ FIXED: Exactly 2 nodes!

    route_flat_list.append(int(d_idx))
    current_route_len += 1
    route_lens_list.append(current_route_len)

    while len(route_lens_list) < int(n_veh):
        route_flat_list.extend([int(d_idx), int(d_idx)])
        route_lens_list.append(2)

    route_flat_arr = np.array(route_flat_list, dtype=np.int64)
    route_lens_arr = np.array(route_lens_list, dtype=np.int64)

    obj, t_time, d_m, s_time, w_time, tw_p, lat = evaluate_routes_kernel(
        route_flat_arr, route_lens_arr, demands, tw_starts, tw_ends, service_times,
        dist_matrix, time_matrix, float(v_cap), float(t_wt), float(d_wt),
        float(disp_s), float(lat_pen)
    )
    return float(obj), route_flat_list

class QPSO:
    def __init__(self, evaluate=None, n_particles=40, iterations=100, seed=42, max_workers=None,
                 dist_matrix=None, time_matrix=None, demands=None,
                 tw_starts=None, tw_ends=None, service_times=None, vehicle_capacity=None,
                 num_vehicles=None, depot_idx=0, time_weight=1.0, distance_weight=0.0):
        self.evaluate = evaluate
        self.n_particles = n_particles
        self.iterations = iterations
        self.max_workers = max_workers or min(os.cpu_count() or 2, 8)
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
        particles = self.rng.uniform(0, 1, (self.n_particles, dimensions))

        if initial_population is not None:
            init_pop = np.asarray(initial_population, dtype=float)
            n_init = min(len(init_pop), self.n_particles)
            particles[:n_init] = np.clip(init_pop[:n_init], 0.0, 1.0)
        elif initial_position is not None:
            initial_position = np.asarray(initial_position, dtype=float)
            particles[0] = np.clip(initial_position, 0.0, 1.0)

        pbest = particles.copy()

        worker_args_template = (
            self.dist_matrix, self.time_matrix, 
            self.demands, self.tw_starts, self.tw_ends, self.service_times, 
            float(self.vehicle_capacity), int(self.num_vehicles), int(self.depot_idx),
            float(self.time_weight), float(self.distance_weight), 32400.0, 10000.0
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            initial_args = [(particles[i],) + worker_args_template for i in range(self.n_particles)]
            initial_results = list(executor.map(_thread_evaluate, initial_args))

            pbest_scores = np.array([res[0] for res in initial_results])
            gbest_idx = np.argmin(pbest_scores)
            gbest = pbest[gbest_idx].copy()
            gbest_score = pbest_scores[gbest_idx]

            history = [gbest_score]

            for it in range(self.iterations):
                progress = it / max(1, self.iterations - 1)
                alpha = 1.0 - 0.8 * progress

                sorted_indices = np.argsort(pbest_scores)
                n_elite = max(5, int(0.3 * self.n_particles))
                elite_indices = sorted_indices[:n_elite]
                elite_pbest = pbest[elite_indices]
                elite_scores = pbest_scores[elite_indices]

                score_diffs = elite_scores - elite_scores[0]
                exp_scores = np.exp(-score_diffs / (np.ptp(elite_scores) + 1e-8))
                weights = exp_scores / np.sum(exp_scores)
                mbest = np.sum(elite_pbest * weights[:, np.newaxis], axis=0)

                for i in range(self.n_particles):
                    phi = self.rng.uniform(0, 1, dimensions)
                    p = phi * pbest[i] + (1 - phi) * gbest
                    u = np.maximum(self.rng.uniform(0, 1, dimensions), np.finfo(float).tiny)
                    sign = np.where(self.rng.random(dimensions) > 0.5, 1, -1)
                    particles[i] = p + sign * alpha * np.abs(mbest - particles[i]) * np.log(1.0 / u)
                    particles[i] = np.clip(particles[i], 0.0, 1.0)

                iter_args = [(particles[i],) + worker_args_template for i in range(self.n_particles)]
                results = list(executor.map(_thread_evaluate, iter_args))

                for i, (score, _) in enumerate(results):
                    if score < pbest_scores[i]:
                        pbest[i] = particles[i].copy()
                        pbest_scores[i] = score
                        if score < gbest_score:
                            gbest = particles[i].copy()
                            gbest_score = score

                history.append(gbest_score)

        return gbest, gbest_score, history