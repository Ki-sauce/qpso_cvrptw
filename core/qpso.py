import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

class QPSO:
    def __init__(self, evaluate, n_particles=40, iterations=100, seed=42, max_workers=None):
        if n_particles < 1 or iterations < 1:
            raise ValueError("QPSO requires positive particles and iterations.")
        self.evaluate = evaluate
        self.n_particles = n_particles
        self.iterations = iterations
        self.max_workers = max_workers or min(4, os.cpu_count() or 2)
        self.rng = np.random.default_rng(seed)
        
    def optimize(self, dimensions, initial_position=None, initial_population=None):
        # Initialize particles in continuous random-key space [0, 1]^N (matching GA)
        particles = self.rng.uniform(0, 1, (self.n_particles, dimensions))
        
        # Support full multi-particle population priming from ML warm-start
        if initial_population is not None:
            init_pop = np.asarray(initial_population, dtype=float)
            n_init = min(len(init_pop), self.n_particles)
            particles[:n_init] = np.clip(init_pop[:n_init], 0.0, 1.0)
        elif initial_position is not None:
            initial_position = np.asarray(initial_position, dtype=float)
            if initial_position.shape != (dimensions,):
                raise ValueError("initial_position has the wrong dimension.")
            particles[0] = np.clip(initial_position, 0.0, 1.0)
            
        pbest = particles.copy()
        
        # Parallel evaluation of initial swarm population
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            initial_results = list(executor.map(self.evaluate, particles))
        
        pbest_scores = np.array([res[0] for res in initial_results])
        gbest_idx = np.argmin(pbest_scores)
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
                phi = self.rng.uniform(0, 1, dimensions)
                p = phi * pbest[i] + (1 - phi) * gbest
                
                # Quantum potential well position update
                u = np.maximum(
                    self.rng.uniform(0, 1, dimensions),
                    np.finfo(float).tiny,
                )
                sign = np.where(self.rng.random(dimensions) > 0.5, 1, -1)
                particles[i] = p + sign * alpha * np.abs(mbest - particles[i]) * np.log(1.0 / u)
                
                # Boundary clamping to [0, 1]
                particles[i] = np.clip(particles[i], 0.0, 1.0)
            
            # --- SWARM DIVERSITY INJECTION (JITTER) ---
            # If standard deviation across particles drops too low, re-initialize 
            # 20% of the swarm to escape local optima traps.
            diversity = np.mean(np.std(particles, axis=0))
            if diversity < 0.008 and it < self.iterations * 0.8:
                n_reset = max(1, int(0.2 * self.n_particles))
                # Never reset the global best index
                reset_indices = [idx for idx in self.rng.choice(self.n_particles, n_reset, replace=False) if idx != gbest_idx]
                for ri in reset_indices:
                    particles[ri] = self.rng.uniform(0, 1, dimensions)
                    pbest[ri] = particles[ri].copy()
            # -------------------------------------------
            
            # Evaluate all updated particle positions concurrently in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.evaluate, particles))
            
            # Process results and update personal/global bests
            for i, (score, _) in enumerate(results):
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score
                    if score < gbest_score:
                        gbest = particles[i].copy()
                        gbest_score = score
                        
            history.append(gbest_score)
            
        return gbest, gbest_score, history