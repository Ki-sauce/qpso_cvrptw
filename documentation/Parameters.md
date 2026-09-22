## I. System Parameters (Environment Setup)

These parameters define the physical and logistical constraints of the routing problem instance.

| Parameter | Type | Description | Technical Implementation |
| --- | --- | --- | --- |
| **OSM Location** | `String` | Target administrative boundary for the road network. | Passed to `osmnx.graph_from_place()` to download the geometric multigraph. |
| **Customers** | `Integer` | Number of delivery destinations ($N$). | Determines the dimensionality of the QPSO search space ($\mathbb{R}^N$). |
| **Vehicles** | `Integer` | Maximum available fleet size ($K$). | Caps the maximum number of sub-routes decoded during objective evaluation. |
| **Vehicle Capacity** | `Float` | Maximum load limit per vehicle ($C$). | Triggers a new vehicle deployment during route decoding if the next customer demand exceeds remaining capacity. |
| **Time-Windows** | `Customer fields` | Ready time, due time, and service duration are evaluated for every route. | Early arrivals wait; late service starts receive the shared lateness penalty. |
| **Distance Weight** | `Float` | The $\beta$ scalar in the linear objective function. | Evaluates total cost as `(Time_s * 1.0) + (Distance_m * Distance_Weight)`. |

## II. QPSO Hyperparameters

These variables strictly govern the behavior, exploration, and convergence of the metaheuristic engine.

| Hyperparameter | Type | Description | Mathematical Impact |
| --- | --- | --- | --- |
| **Swarm Particles** | `Integer` | Population size of the swarm ($S$). | Dictates the number of concurrent quantum states evaluated per iteration. Larger swarms increase global exploration but scale execution time linearly. |
| **Max Iterations** | `Integer` | Termination criteria ($T_{max}$). | Determines the decay rate of the Contraction-Expansion (CE) coefficient. |
| **CE Coefficient Base** | `Float` | Expansion scale ($\gamma$). | Implemented internally. Decays linearly from 1.0 to 0.5 to transition from global search to local exploitation. |

## III. Stochasticity, Distributions, and Seeding

The framework relies heavily on controlled stochasticity for both environment generation and QPSO mechanics. A single global integer, `traffic_seed`, is passed to `numpy.random.default_rng(seed)` across all modules to guarantee exact reproducibility for scientific benchmarking.

* **Node Selection:** The depot and $N$ customer nodes are selected via uniform random sampling (`rng.choice`) without replacement from the graph's largest connected component.
* **Customer Demand Generation:** Modeled using a discrete uniform distribution. Each customer is assigned a demand unit via `rng.integers(1, 8)`.
* **Dynamic Traffic Congestion:** Matching edges use the generated BPR CSV. Unmatched locations use a deterministic seed-controlled fallback multiplier, with rush-hour and off-peak modes.
* **QPSO Local Attractor ($\phi$):** During iteration, the focal point between a particle's personal best and the global best is determined by $\phi \sim U(0, 1)$.
* **QPSO Wavefunction Collapse ($u$):** The Monte Carlo sampling used to collapse the quantum state into a discrete position utilizes $u \sim U(0, 1)$ to scale the natural logarithm function.
* **Swarm Initialization:** The initial positions of all particles in $\mathbb{R}^N$ are seeded using a continuous uniform distribution bounded by the problem dimensionality.

## IV. Technical Pipeline Architecture

The framework is decoupled into a modular pipeline to separate geospatial graph processing from combinatorial mathematics.

* `app.py`: Handles state management, UI rendering, and orchestration of the execution pipeline. Caches the graph layout to prevent redundant OSMnx API calls.
* `core/graph_model.py`: Fetches and projects the physical road geometry from OpenStreetMap into a NetworkX multidigraph and initializes missing edge metrics.
* `core/heuristics.py`: Contains capacity- and time-window-aware greedy construction with Dijkstra or A*. A* uses a distance-to-time lower bound scaled by the graph's maximum observed speed.
* `core/qpso.py`: The isolated mathematics engine. Operates entirely in a continuous topological space.
* `core/vrp.py`: Contains the **Random-Key Decoder**. Because QPSO outputs a continuous position vector $X \in \mathbb{R}^N$, this module applies an algebraic `argsort()` to convert the vector into a discrete customer visitation sequence, splits it into multiple vehicle routes based on capacity constraints, and scores it against the shortest-path matrix.
