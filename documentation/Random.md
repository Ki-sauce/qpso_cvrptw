## Randomness Documentation

Node selection, customer demand, time-window offsets, QPSO, GA, and unmatched-edge traffic all originate from the requested master seed.

Using a single master seed is standard scientific practice for Operations Research. It ensures "Global Reproducibility"—if a judge runs your app with Seed 42, they will see the exact same customers, the exact same traffic jams, and the exact same QPSO convergence curve you saw.

Here is how the randomness is partitioned under the hood, and how you can swap the distributions to model different real-world scenarios.

**1. Customer & Depot Locations**

* **Current Distribution:** Uniform Random Choice without replacement.
* **Mechanism:** `rng.choice(nodes, size=n+1, replace=False)`. Every intersection in the city has an equal $1/N$ probability of being picked.
* **How to Change It:** If you want to model a dense downtown delivery zone with a few sparse suburban outliers, you would cluster them using a 2D Gaussian (Normal) distribution around a specific coordinate, snapping the results to the nearest OSM graph nodes using `ox.distance.nearest_nodes()`.

**2. Customer Demand**

* **Current Distribution:** Discrete Uniform Distribution.
* **Mechanism:** `rng.integers(1, 8, size=N)`. A customer is equally likely to order 1, 2, 3, 4, 5, 6, or 7 items.
* **How to Change It:** Real-world logistics rarely follow a uniform distribution. Most people order small packages, and a few order massive freight. You should change this to a **Poisson Distribution** (standard for arrival/demand modeling):
```python
# lambda (lam) is the average demand.
demands = rng.poisson(lam=3.0, size=len(customer_nodes))
# Ensure no zero-demand customers
demands = np.clip(demands, 1, capacity) 

```



**3. Traffic Congestion**

* **Current Distribution:** Generated BPR edge records when the traffic dataset matches the graph; otherwise a seeded clipped normal fallback is used.
* **Mechanism:** `simulated`/`off_peak` and `live`/`rush_hour` select the corresponding traffic scenario. The latter is a deterministic mock, not a live API.
* **How to Change It:** Replace the CSV loader with a provider-specific traffic adapter while keeping `travel_time_s` and `distance_m` on each edge.



**4. QPSO Wavefunction (Algorithm Randomness)**

* **Current Distribution:** Continuous Uniform $U(0,1)$.
* **Mechanism:** `u = rng.uniform(0.0, 1.0)` and `phi = rng.uniform(0.0, 1.0)`.
* **How to Change It:** **Do not change this.** The mathematical proof for QPSO's global convergence relies entirely on uniformly sampling the inverse cumulative distribution function of the Laplace distribution. Changing `u` or `phi` to a Gaussian will break the quantum delta-potential well mechanics and destroy the algorithm's ability to search the space properly.
