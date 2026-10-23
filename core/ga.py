import numpy as np
from .jit_eval import evaluate_swarm_jit

def _order_crossover(parent1, parent2, rng):
    """Permutation-preserving order crossover (OX)."""
    n = len(parent1)
    if n < 2:
        return parent1.copy(), parent2.copy()
    left, right = sorted(rng.choice(n, 2, replace=False))

    def make_child(first, second):
        child = np.full(n, -1, dtype=int)
        child[left:right + 1] = first[left:right + 1]
        used = set(child[left:right + 1])
        insert_at = (right + 1) % n
        for value in np.concatenate((second[right + 1:], second[:right + 1])):
            if value not in used:
                child[insert_at] = value
                used.add(int(value))
                insert_at = (insert_at + 1) % n
        return child

    return make_child(parent1, parent2), make_child(parent2, parent1)


def _mutate_swap(permutation, rng):
    if len(permutation) > 1:
        left, right = rng.choice(len(permutation), 2, replace=False)
        permutation[left], permutation[right] = permutation[right], permutation[left]


def run_ga(
    jit_args,
    dimensions,
    population_size=40,
    iterations=100,
    mutation_rate=0.1,
    seed=42,
    initial_permutation=None,
):
    """Permutation-aware genetic algorithm using the unified Numba JIT evaluator."""
    if dimensions < 1:
        raise ValueError("GA requires at least one dimension.")
    if population_size < 3 or iterations < 1:
        raise ValueError("GA requires at least three individuals and one iteration.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between zero and one.")

    rng = np.random.default_rng(seed)
    
    population = np.array(
        [rng.permutation(dimensions) for _ in range(population_size)],
        dtype=int,
    )
    if initial_permutation is not None:
        initial_permutation = np.asarray(initial_permutation, dtype=int)
        if sorted(initial_permutation.tolist()) != list(range(dimensions)):
            raise ValueError("initial_permutation must be a valid permutation.")
        population[0] = initial_permutation
        
    global_best_cost = np.inf
    global_best_position = None
    history = []

    for _ in range(iterations):
        # Map discrete integer permutations to evenly spaced continuous keys.
        # Calling argsort twice inverts the sequence perfectly into continuous [0...1] space.
        continuous_pop = np.argsort(np.argsort(population, axis=1), axis=1).astype(np.float32) / max(1, dimensions - 1)
        
        # Pure C/Numba evaluation (No GIL, single batch array pass)
        fitness_scores = evaluate_swarm_jit(continuous_pop, *jit_args)
        
        best_idx = int(np.argmin(fitness_scores))
        if global_best_position is None or fitness_scores[best_idx] < global_best_cost:
            global_best_cost = float(fitness_scores[best_idx])
            global_best_position = population[best_idx].copy()
        history.append(global_best_cost)

        def tournament():
            competitors = rng.choice(population_size, 3, replace=False)
            return population[competitors[np.argmin(fitness_scores[competitors])]]

        new_population = [global_best_position.copy()]
        while len(new_population) < population_size:
            parent1, parent2 = tournament().copy(), tournament().copy()
            child1, child2 = _order_crossover(parent1, parent2, rng)
            if rng.random() < mutation_rate:
                _mutate_swap(child1, rng)
            if rng.random() < mutation_rate:
                _mutate_swap(child2, rng)
            new_population.extend((child1, child2))
        population = np.array(new_population[:population_size], dtype=int)

    return global_best_position, global_best_cost, history