## Benchmarks

The prototype currently contains:

1. QPSO
2. Random Search baseline
3. Exhaustive exact enumeration for small instances

For the dissertation/project report, add at least one stronger conventional metaheuristic such as:

- Genetic Algorithm
- Simulated Annealing
- classical PSO with a proper permutation decoder

Do not compare QPSO against the broken "continuous PSO + argsort" formulation from the original prototype. The decoder must be identical across algorithms.

Start with 6-8 customers. Exact enumeration becomes expensive very quickly.

## 6. Recommended experiment

For each customer count:

```text
N = 5, 8, 12, 16, 20
```

run:

```text
QPSO
GA
SA
Exact (small N only)
```

Use 20 independent random seeds.

Report:

- best objective
- mean objective
- standard deviation
- median objective
- runtime
- convergence iteration
- optimality gap where exact solution exists
- distance
- travel time
- scalability with number of customers
- performance under low/medium/high congestion

Do not claim "quantum advantage". This is quantum-inspired classical optimization. The defensible claim is improved empirical search behaviour under the tested instances.
