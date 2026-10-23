import numpy as np
from .jit_eval import evaluate_swarm_jit

class QPSO:
    def __init__(self, jit_args, n_particles=40, iterations=100, seed=42):
        if n_particles < 1 or iterations < 1:
            raise ValueError("QPSO requires positive particles and iterations.")
        # Store the flattened numpy arrays to feed the LLVM kernel
        self.jit_args = jit_args
        self.n_particles = n_particles
        self.iterations = iterations
        self.rng = np.random.default_rng(seed)
        
    def optimize(self, dimensions, initial_position=None, initial_population=None):
        # Initialize particles in continuous random-key space [0, 1]^N
        particles = self.rng.uniform(0, 1, (self.n_particles, dimensions)).astype(np.float32)
        
        # Support full multi-particle population priming from ML warm-start
        if initial_population is not None:
            init_pop = np.asarray(initial_population, dtype=np.float32)
            n_init = min(len(init_pop), self.n_particles)
            particles[:n_init] = np.clip(init_pop[:n_init], 0.0, 1.0)
        elif initial_position is not None:
            initial_position = np.asarray(initial_position, dtype=np.float32)
            if initial_position.shape != (dimensions,):
                raise ValueError("initial_position has the wrong dimension.")
            particles[0] = np.clip(initial_position, 0.0, 1.0)
            
        pbest = particles.copy()
        
        # 1. Pure C/Numba evaluation (No GIL, single batch array pass)
        pbest_scores = evaluate_swarm_jit(particles, *self.jit_args)
        
        gbest_idx = int(np.argmin(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        history = [gbest_score]
        
        for it in range(self.iterations):
            # Extended contraction-expansion schedule for deeper exploration phase
            progress = it / max(1, self.iterations - 1)
            alpha = 1.0 - 0.8 * progress
            
            # Compute Mean Best Position (mbest) of the swarm
            mbest = np.mean(pbest, axis=0)
            
            # Update all particle positions vector-wise first
            for i in range(self.n_particles):
                # Quantum attractor: stochastic blend of personal and global best
                phi = self.rng.uniform(0, 1, dimensions).astype(np.float32)
                p = phi * pbest[i] + (1.0 - phi) * gbest
                
                # Quantum potential well position update
                u = np.maximum(
                    self.rng.uniform(0, 1, dimensions),
                    np.finfo(np.float32).tiny,
                ).astype(np.float32)
                
                sign = np.where(self.rng.random(dimensions) > 0.5, 1.0, -1.0).astype(np.float32)
                particles[i] = p + sign * alpha * np.abs(mbest - particles[i]) * np.log(1.0 / u)
                
                # THE CROWN JEWEL: Double-Argsort Rank Re-normalization
                # Projects values back into evenly spaced linear ordinal distribution to prevent sequence collapse
                rank_order = np.argsort(particles[i])
                double_rank = np.argsort(rank_order)
                particles[i] = double_rank.astype(np.float32) / float(max(1, dimensions - 1))
            
            # --- SWARM DIVERSITY INJECTION (JITTER) ---
            diversity = np.mean(np.std(particles, axis=0))
            if diversity < 0.008 and it < self.iterations * 0.8:
                n_reset = max(1, int(0.2 * self.n_particles))
                reset_indices = [idx for idx in self.rng.choice(self.n_particles, n_reset, replace=False) if idx != gbest_idx]
                for ri in reset_indices:
                    particles[ri] = self.rng.uniform(0, 1, dimensions).astype(np.float32)
                    pbest[ri] = particles[ri].copy()
            # -------------------------------------------
            
            # 2. Pure C/Numba evaluation (No GIL)
            swarm_scores = evaluate_swarm_jit(particles, *self.jit_args)
            
            # Process results and update personal/global bests
            for i in range(self.n_particles):
                if swarm_scores[i] < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = swarm_scores[i]
                    if swarm_scores[i] < gbest_score:
                        gbest = particles[i].copy()
                        gbest_score = swarm_scores[i]
                        gbest_idx = i
                        
            history.append(gbest_score)
            
        return gbest, gbest_score, history