import itertools
import math
import numpy as np


def random_key_order(position, customers):
    idx = np.argsort(position)
    return [customers[i] for i in idx]


def run_random_search(evaluate, dimension, iterations=1000, seed=42, initial_x=None):
    if dimension < 1 or iterations < 1:
        raise ValueError("Random search requires positive dimension and iterations.")
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = math.inf
    history = []

    for _ in range(iterations):
        x = initial_x.copy() if initial_x is not None and best_x is None else rng.random(dimension)
        score, _ = evaluate(x)
        if best_x is None or score < best_score:
            best_score = score
            best_x = x.copy()
        history.append(best_score)

    return best_x, best_score, history


def run_exact_small(evaluate, customers, max_customers=9):
    """
    Exhaustive permutation benchmark for small instances.
    Deliberately limited because VRP is NP-hard.
    """
    if len(customers) > max_customers:
        raise ValueError(
            f"Exact benchmark limited to <= {max_customers} customers."
        )

    best_order = None
    best_score = math.inf

    # Convert a permutation into a random-key vector whose ordering is that permutation.
    for perm in itertools.permutations(range(len(customers))):
        x = np.empty(len(customers), dtype=float)
        for rank, idx in enumerate(perm):
            x[idx] = rank / max(1, len(customers) - 1)

        score, _ = evaluate(x)
        if score < best_score:
            best_score = score
            best_order = x

    return best_order, best_score


def gap_percent(heuristic, reference):
    if not math.isfinite(reference) or reference == 0:
        return math.nan
    return 100.0 * (heuristic - reference) / reference
