# QPSO CVRPTW Project Guide

This document explains the complete project workflow, mathematical model,
algorithm implementations, file responsibilities, frontend/backend contract,
testing strategy, and extension points.

It is written for two audiences at once:

- someone who wants to understand what the application does in practical terms;
- someone who wants enough technical detail to debug, improve, or extend it.

## 1. What problem does this project solve?

The project solves a **Capacitated Vehicle Routing Problem with Time Windows**
(CVRPTW) on a real road network.

In plain language:

> Given a depot, several customers, a fleet of delivery vehicles, vehicle load
> limits, customer availability windows, and changing traffic conditions, find
> good routes that visit every customer and return to the depot with as little
> operational cost as possible.

Imagine a dispatch manager at 9:00 AM with four delivery vans. Each customer has:

- a location on a road network;
- a quantity that must be delivered;
- a time at which they become ready to receive the delivery;
- a deadline;
- a service/unloading duration.

The manager must decide:

1. which customers each vehicle serves;
2. the order in which each vehicle visits them;
3. which actual roads each vehicle takes between consecutive stops;
4. how traffic affects the travel time;
5. whether the planned service times are respected.

The project compares three strategies for the same routing instance:

- **QPSO**: Quantum-inspired Particle Swarm Optimization;
- **GA**: a permutation-aware Genetic Algorithm;
- **A***: a greedy constructive baseline using A* for individual road legs.

The algorithms are not quantum-computing implementations. QPSO is a classical
stochastic optimizer inspired by quantum-behavior equations.

## 2. Important claim: “optimal” versus “near-optimal”

CVRPTW is NP-hard. As the number of customers grows, exhaustively checking
every possible route order becomes impractical.

Therefore:

- QPSO does **not** mathematically prove a global optimum for large instances.
- GA does not prove a global optimum either.
- A* finds shortest paths for individual graph legs, but its greedy customer
  selection is not a global VRP optimizer.
- `solve_exact` exhaustively checks permutations only for small instances, by
  default up to nine customers, and is the project’s small-instance reference.

QPSO is intended to find a strong, often near-optimal solution in a difficult
search space within a practical computation budget. “Best” always means best
under the configured objective, seed, traffic scenario, particle count, and
iteration count. It is not guaranteed to beat GA or A* on every possible input.

## 3. Repository workflow at a glance

The complete dashboard request follows this pipeline:

```text
Browser dashboard
      |
      | POST /api/optimize
      v
FastAPI request validation
      |
      v
OpenStreetMap drivable road graph
      |
      v
Connected graph cleanup and edge metrics
      |
      v
Traffic scenario applied to each road edge
      |
      v
Deterministic, spatially spread depot and customer selection
      |
      v
Demand generation and fleet-capacity feasibility check
      |
      v
Customer time-window construction
      |
      +-------------------+------------------+
      |                   |                  |
      v                   v                  v
     A*                  GA                QPSO
      |                   |                  |
      +-------------------+------------------+
                          v
                Shared route evaluator
                          |
                          v
              Finite-result and customer-set checks
                          |
                          v
             GeoJSON routes + schedules + metrics
                          |
                          v
                 Map animation and dashboard
```

The Streamlit application uses the same core algorithms, but orchestrates them
locally from `app.py` instead of going through the FastAPI endpoint.

## 4. Running the project

The practical setup instructions are in [run.md](run.md). The two processes used
by the main dashboard are:

```text
FastAPI:  http://localhost:8000
Next.js:  http://localhost:3000
```

The frontend sends requests to FastAPI. Python packages are installed from
`requirements.txt`; JavaScript packages are installed from
`vrp-dashboard/package.json` and `package-lock.json` with `npm ci`.

## 5. Input parameters

The FastAPI request model is `OptimizationRequest` in `core/api.py`.

| Parameter | Meaning | Current validation/default |
| --- | --- | --- |
| `place_name` | OSM place used to download the road network | non-empty; default Salt Lake, Kolkata |
| `customers_n` | Number of delivery stops | 1–500; default 10 |
| `num_vehicles` | Available vehicles | 1–100; default 4 |
| `vehicle_capacity` | Maximum load per vehicle | positive; default 25 |
| `distance_weight` | Weight of physical distance in the objective | non-negative; default 0.2 |
| `particles` | QPSO swarm size and GA population size | 3–1000; default 40 |
| `iterations` | QPSO/GA iteration budget | 1–5000; default 100 |
| `traffic_mode` | Dashboard mode | `simulated` or `live` |
| `traffic_seed` | Reproducibility seed | integer; default 42 |
| `sla_strictness_hours` | Length of each customer time window | greater than 0 and at most 24; default 4 |

In the current dashboard, `live` is a rush-hour/mock scenario selector. It does
not call a live external traffic provider. The traffic data comes from the
bundled dataset when an edge matches it, and from a seeded fallback model when
it does not.

The API uses `traffic_seed` for stop selection, demand generation, and customer
time-window generation. Its current traffic call uses the traffic module's
default fallback seed (`42`) rather than passing the request seed, so unmatched
edge traffic remains the same across API requests with different
`traffic_seed` values. The Streamlit path passes its selected seed to the traffic
function. If per-request fallback traffic variation is required in the API,
change the call in `core/api.py` to pass `seed=req.traffic_seed` and update the
reproducibility tests accordingly.

## 6. The mathematical CVRPTW model

### 6.1 Sets and data

Let:

- (V = \{0,1,\ldots,N\}) be the depot and customers;
- (0) represent the depot;
- (C = V \setminus \{0\}) be the customer set;
- (K = \{1,\ldots,K\}) be the vehicle set;
- (d_i) be the demand of customer (i);
- (Q) be the capacity of each vehicle;
- (t_{ij}) be the traffic-adjusted shortest-path travel time from stop (i) to stop (j);
- (l_{ij}) be the corresponding physical distance;
- ([a_i,b_i]) be customer (i)'s ready/due time window;
- (s_i) be the service time at customer (i).

The road network is not a complete Euclidean graph. The route from (i) to
(j) is a shortest path through actual road nodes and edges.

### 6.2 Conceptual binary formulation

A traditional formulation might use:

```text
x[i,j,k] = 1 if vehicle k travels directly from stop i to stop j
y[i,k]   = 1 if vehicle k serves customer i
T[i,k]   = service start time at customer i for vehicle k
```

Typical constraints are:

```text
Each customer is served once:
    sum(k) y[i,k] = 1                         for every customer i

Vehicle capacity:
    sum(i) demand[i] * y[i,k] <= Q             for every vehicle k

Vehicle starts and finishes at depot:
    every active route begins at 0 and ends at 0

Time propagation:
    T[j,k] >= T[i,k] + service[i] + t[i,j]

Time windows:
    ready[i] <= T[i,k] <= due[i]
```

The real implementation avoids directly optimizing thousands of binary edge
variables. Instead it uses a permutation/random-key decoder described below.
This reduces the optimizer’s representation to one vector of length (N).

### 6.3 Actual objective used by the code

`evaluate_routes` in `core/vrp.py` computes:

```text
objective =
    time_weight * (travel_time + waiting_time + service_time)
  + distance_weight * distance
  + lateness_penalty
```

The current solvers use `time_weight = 1.0` and the dashboard’s
`distance_weight` value.

Lateness is calculated as:

```text
lateness_penalty = (total_lateness_seconds / 60) * 10000
```

This means one late minute is intentionally very expensive compared with one
second of travel. The units are an optimization score, not rupees.

The evaluator handles a customer as follows:

```text
arrival       = current_time + travel_time
service_start = max(arrival, ready_time)
waiting       = max(0, ready_time - arrival)
lateness      = max(0, service_start - due_time)
current_time  = service_start + service_time
```

In the current model, each route starts at 09:00 AM (`32400` seconds). Early
arrival creates vehicle waiting time. Late arrival creates SLA lateness. The
dashboard’s “SLA Penalty” refers to lateness, not early-arrival waiting.

## 7. Why a permutation decoder is used

QPSO naturally operates on continuous vectors, while VRP decisions are discrete.
The project bridges this mismatch using **random-key encoding**.

Suppose a particle is:

```text
X = [0.71, 0.13, 0.94, 0.31]
```

The values correspond to customers `[C1, C2, C3, C4]`. Sorting the values gives:

```text
C2, C4, C1, C3
```

That sorted order becomes the visit order.

Why this is useful:

- QPSO can update real-valued coordinates;
- `argsort` converts the coordinates into a valid customer permutation;
- the decoder handles vehicle splitting and capacity;
- every optimizer can use the same representation and evaluator.

Real-world analogy: each customer receives a priority card. The smallest number
gets visited first. QPSO moves the numbers around; whenever their order changes,
the delivery sequence changes.

An important consequence is that many nearby continuous vectors can decode to the
same permutation. A QPSO movement only changes the route when values cross their
ranking boundaries. This is normal for random-key optimization, but it is also a
future improvement area if finer discrete neighborhood control is needed.

## 8. Road-graph construction and shortest paths

### 8.1 `core/api.py`: production API graph

`prepare_osm_graph(place_name)` calls OSMnx to download a drivable, simplified
OpenStreetMap network in WGS84 coordinates.

The function then:

1. keeps the largest strongly connected component for directed graphs;
2. keeps the largest connected component for undirected graphs;
3. copies the subgraph;
4. ensures every edge has `distance_m`;
5. ensures every edge has `travel_time_s`;
6. caches the result in the process-local `GRAPH_CACHE`.

Keeping one connected component is important: a selected customer should have a
legal road path to every other selected stop.

### 8.2 Edge objective weights

`add_objective_weights` creates a copy of the graph and assigns every edge:

```text
_routing_weight = time_weight * travel_time_s
                + distance_weight * distance_m
```

This is the weight used to choose routes. The original travel time and physical
distance remain available for reporting.

### 8.3 Stop-to-stop matrix

`build_stop_matrix` runs NetworkX Dijkstra from every depot/customer stop to all
other selected stops. It records:

- weighted shortest-path cost;
- the actual graph-node path.

If there are (N+1) selected stops, this creates an ((N+1) \times (N+1))
lookup. Optimizers then evaluate route orders using this matrix instead of
re-running a full road search for every particle and every iteration.

This is a major performance decision: the expensive physical network search is
precomputed once per optimization request.

`path_metrics` converts a path back into physical metrics by summing the selected
edges’ travel time and distance. It correctly handles parallel edges in a
NetworkX `MultiDiGraph`.

## 9. Spatial stop selection

`core/stop_selection.py` contains `choose_spread_stops`.

Earlier random node selection could choose many nearby graph nodes simply because
dense neighbourhoods contain more road nodes. The current method:

1. finds the largest connected component;
2. removes nodes without valid `x/y` coordinates;
3. chooses a seeded first node;
4. repeatedly selects one of the farthest candidates from the already-selected set;
5. returns one depot and the requested number of customers.

This is a seeded farthest-point strategy. It expands the geographical coverage
of the delivery locations while preserving reproducibility.

For very large networks, the candidate pool is capped at 12,000 nodes to prevent
stop selection from dominating the request time.

## 10. Traffic workflow

### 10.1 Bundled traffic data

`core/traffic_dataset.csv` stores edge-level traffic observations. The loader in
`core/traffic.py` builds two lookups:

- `(u, v)` pair records;
- `(u, v, key)` records for parallel edges.

Each record includes off-peak and rush-hour travel times and speeds.

### 10.2 Applying a scenario

`apply_traffic_scenario` copies the graph and updates each edge.

The dashboard maps:

```text
simulated -> off_peak
live      -> rush_hour/mock
```

For a matching CSV edge, the corresponding traffic time is applied. For an
unmatched edge, the code retains a baseline travel time or estimates one from a
default speed, then applies a deterministic seeded multiplier. This fallback is
why a location not represented in the bundled CSV can still be processed.

The resulting `travel_time_s` values are what Dijkstra, A*, GA, and QPSO see.

### 10.3 Generating a new traffic dataset

`core/generate_traffic_data.py` downloads a graph and generates traffic records.
It uses:

- road-class free-flow speeds;
- randomly generated congestion ratios from Beta distributions;
- spatial hotspot effects;
- the Bureau of Public Roads-style formula:

```text
T = T_free_flow * (1 + 0.15 * (V/C)^4)
```

It writes the resulting CSV to the `core` directory by default.

This script is useful when the selected area or graph IDs differ significantly
from the bundled dataset. Always regenerate the dataset with the same graph
source and compatible node/edge identifiers.

## 11. The A* constructive baseline

`core/heuristics.py` implements `solve_dynamic_heuristic`.

### 11.1 Road-leg routing

For each candidate leg:

- `method="astar"` uses `networkx.astar_path`;
- any other method uses `networkx.dijkstra_path`.

The A* heuristic estimates a lower-bound travel objective using:

```text
straight-line distance / maximum observed graph speed
```

For geographic coordinates, straight-line distance uses a haversine calculation.
The heuristic is intended to be admissible when the edge metrics obey the same
physical units and assumptions.

### 11.2 Greedy customer construction

The baseline starts at the depot and repeatedly chooses the unvisited customer
with the lowest local candidate score among customers that fit the current
vehicle’s remaining capacity.

The candidate score includes:

```text
leg objective
+ time-weighted early-arrival waiting
+ time-weighted service time
+ large lateness penalty
```

When no more customer fits, the vehicle returns to the depot and the next vehicle
starts. This is a local, one-step decision. It does not evaluate all future
customer combinations, which is why it is a baseline rather than a global VRP
optimizer.

### 11.3 Completeness repair

A greedy bin-packing order can strand a customer even when the complete demand
set is feasible. The implementation detects remaining `unvisited` customers and
repairs the order using `construct_feasible_order` plus the same A*/Dijkstra leg
router.

The API additionally verifies that every algorithm:

- serves the same customer node set;
- serves each customer exactly once;
- produces finite route metrics.

This prevents the UI from comparing QPSO on all customers with a partial A* or GA
solution.

## 12. The Genetic Algorithm

`core/ga.py` implements a permutation-aware GA.

### 12.1 Representation

Each individual is an integer permutation such as:

```text
[2, 0, 3, 1]
```

The shared engine converts that permutation into customer order and then uses the
same capacity decoder and route evaluator as QPSO.

### 12.2 Evolution steps

Each generation performs:

1. evaluate every permutation;
2. keep the best individual through elitism;
3. select parents using three-way tournament selection;
4. create children with order crossover (OX);
5. optionally apply swap mutation;
6. continue until the requested population size is restored.

Order crossover is permutation-preserving: it never creates duplicate or missing
customer indices.

GA is useful as a classical metaheuristic comparison. It searches directly in
permutation space, while QPSO searches continuous random-key space and decodes it
to permutations.

## 13. QPSO in this project

`core/qpso.py` contains the QPSO mathematics. `core/engine.py` connects it to
the VRP representation and evaluator.

### 13.1 Particle state

For (N) customers, every particle is a vector:

```text
X = [x1, x2, ..., xN],    0 <= xi <= 1
```

The vector is not a direct route. Its sorted order is the route permutation.

Each particle stores:

- its current position `particles[i]`;
- its personal best position `pbest[i]`;
- its personal best score `pbest_scores[i]`.

The swarm stores:

- global best position `gbest`;
- global best objective `gbest_score`;
- convergence history.

### 13.2 Initialization

The swarm starts with uniform random values in `[0,1]`.

The engine also constructs one capacity-feasible order and injects its random-key
representation into particle zero. This gives the swarm at least one strong,
valid starting point when the fleet is physically feasible.

All random generation uses `numpy.random.default_rng(seed)`, so a fixed seed,
graph, traffic scenario, and parameter set produce reproducible behavior.

### 13.3 Quantum-inspired update

At each iteration, QPSO computes:

```text
mbest = mean(pbest positions)
```

For each particle and dimension, it draws:

```text
phi ~ Uniform(0, 1)
u   ~ Uniform(0, 1)
sign in {-1, +1}
```

Then it builds a local attractor:

```text
p = phi * pbest + (1 - phi) * gbest
```

The new position is:

```text
X_new = p + sign * alpha * |mbest - X| * ln(1/u)
```

The implementation uses:

```text
alpha = 1.0 - 0.5 * progress
```

where `progress` moves from 0 to 1 over the iteration budget. Thus `alpha`
contracts from 1.0 to 0.5:

- early iterations make larger exploratory jumps;
- later iterations concentrate around promising regions.

The vector is clipped back into `[0,1]` after every update.

### 13.4 Selection and convergence

The updated particle is evaluated. If its score improves its personal best, the
personal best is replaced. If it improves the global best, the global best is
replaced.

The returned result is:

- the best random-key vector found;
- its objective score;
- the best-score history over iterations.

### 13.5 QPSO route evaluation and 2-opt

`solve_qpso` calls the shared evaluator with `apply_quantum_annealing=True`.
Despite the name, the implementation is a deterministic classical 2-opt local
improvement applied to each decoded route:

1. select a route segment;
2. reverse it;
3. keep the reversal if its cached weighted stop-to-stop cost improves.

This provides local exploitation after QPSO proposes a global ordering. Capacity
is preserved because 2-opt only changes the order inside an existing vehicle
route. Time-window effects are still evaluated afterward by the shared scorer.

### 13.6 Why QPSO can work well in this NP-hard space

The full route decision has a combinatorial number of possibilities. QPSO helps
because:

- random keys avoid direct binary edge-variable optimization;
- the swarm explores many route orderings in parallel;
- the quantum-inspired logarithmic jump can escape local regions;
- `mbest` captures the center of the swarm’s personal-best knowledge;
- `gbest` pulls particles toward the best known route;
- the contraction schedule balances exploration and exploitation;
- 2-opt removes obvious local ordering inefficiencies;
- the graph shortest-path matrix makes repeated scoring practical.

However, continuous random-key decoding creates large regions with identical
permutations, and a finite swarm can miss the best ordering. Increasing particles,
iterations, or running multiple seeds improves confidence but increases runtime.

## 14. Shared optimizer engine

`core/engine.py` is the bridge between graph data and optimization algorithms.

### `build_optimizer`

This function:

1. validates objective weights;
2. adds `_routing_weight` to graph edges;
3. builds the all-pairs selected-stop matrix;
4. extracts physical travel time and distance for each stored path;
5. returns an evaluator closure shared by QPSO, GA, random search, and exact search.

The evaluator accepts either:

- an integer permutation, used by GA/exact search;
- a continuous random-key vector, used by QPSO/random search.

It then decodes the order, splits it into routes, optionally applies QPSO’s 2-opt
step, and calls `evaluate_routes`.

### Result payload

`_result_payload` standardizes algorithm output with:

- algorithm name;
- score and history;
- route lists;
- physical travel time;
- distance;
- service time;
- waiting time;
- total duration;
- time-window penalty and lateness;
- cached graph paths.

## 15. Capacity and route decoding

### Feasibility check

`capacity_feasible` checks whether demands can fit into the fleet.

- Up to 24 customers: bounded exact bin-packing search.
- More than 24 customers: deterministic first-fit-decreasing approximation to
  avoid exponential runtime.

For large instances, a `True` result is based on the approximation and is not a
mathematical proof of bin-packing feasibility in every case.

### Feasible order construction

`construct_feasible_order` builds an order whose sequential decoder can use the
fleet.

For small cases it reconstructs an actual exact packing assignment. For large
cases it uses a deterministic first-fit-decreasing assignment.

### Sequential decoder

`split_random_key_solution` walks through the customer order:

```text
if the next demand fits the current vehicle:
    append the customer
else:
    close the current route at the depot
    start the next route
```

Every returned route has the form:

```text
[depot, customer_a, customer_b, ..., depot]
```

Unused fleet slots are represented as `[depot, depot]`.

This decoder is intentionally simple and fast. It does not solve the route
partitioning problem optimally for every possible permutation; the optimizer must
learn customer orders that decode well.

## 16. API request and response workflow

`core/api.py` exposes:

```text
POST /api/optimize
```

The endpoint:

1. validates the request through Pydantic;
2. loads/caches the OSM graph;
3. applies the traffic scenario;
4. chooses spatially spread stops;
5. generates seeded demands and time windows;
6. rejects physically infeasible fleet configurations;
7. runs A*, GA, and QPSO on the same `VRPInstance`;
8. rejects non-finite metrics;
9. verifies identical customer coverage across algorithms;
10. serializes all algorithm routes and schedules;
11. returns the dashboard payload.

The response includes:

```text
locations
business_impact
algorithms
routes                    # QPSO compatibility field
schedules                 # QPSO compatibility field
routes_by_algorithm
schedules_by_algorithm
```

`routes_by_algorithm` and `schedules_by_algorithm` allow the frontend’s algorithm
selector to inspect QPSO, GA, or A* without silently displaying QPSO data for all
three choices.

### GeoJSON route serialization

For every vehicle route, the API concatenates the actual OSM graph paths between
consecutive selected stops into a GeoJSON `LineString`.

The feature properties include:

- vehicle number;
- route color;
- ordered stop labels.

The schedule serializer also records stop IDs, arrival times, depot status, and
total route duration.

## 17. Frontend workflow

### `vrp-dashboard/app/page.tsx`

This is the dashboard coordinator. It:

- stores UI inputs;
- chooses Salt Lake Sector V or Manhattan;
- sends `/api/optimize` requests;
- stores the response;
- lets the user select which algorithm to inspect;
- renders map, schedule, benchmark, and business-impact views.

### `MapViewport.tsx`

This client-only component:

- initializes MapLibre;
- loads Stadia raster tiles;
- renders route GeoJSON layers;
- renders source, intermediate, and destination nodes;
- animates vehicle movement along route geometry;
- highlights the travelled prefix dynamically;
- supports route playback speed and vehicle filtering;
- fits routes to the map;
- hosts draggable/resizable map panels.

The route overlay includes a direct SVG projection fallback in addition to
MapLibre layers. This makes route and node visibility more robust across browser,
style, and raster-basemap conditions.

### `TimelineGantt.tsx`

This renders the returned schedule timelines, including transit blocks, service
blocks, stop labels, arrival times, and late-stop styling.

### `DraggablePanel.tsx`

This is the shared interaction primitive for floating panels. It supports:

- pointer drag from the panel header;
- viewport/container bounds;
- automatic position correction when the sidebar changes size;
- native card resizing where the parent enables `resize`.

## 18. Explanation of every file in `core`

### `core/api.py`

FastAPI application and main production orchestration layer. It owns the request
schema, graph caching, stop generation, algorithm execution, validation, GeoJSON
serialization, schedules, and response contract.

### `core/baselines.py`

Contains generic comparison helpers:

- random-key random search;
- exhaustive small-instance permutation search;
- percentage-gap calculation.

`run_exact_small` is important for validating metaheuristics on tiny instances.

### `core/engine.py`

Shared optimizer adapter. It constructs the graph stop matrix, decodes optimizer
positions, evaluates routes, invokes QPSO/GA, and creates standardized results.

### `core/ga.py`

Permutation-aware Genetic Algorithm primitives: order crossover, swap mutation,
tournament selection, elitism, population evolution, and history tracking.

### `core/generate_traffic_data.py`

Offline traffic-data generator. It downloads an OSM graph, estimates free-flow
road speeds, creates congestion ratios and hotspots, applies a BPR-like formula,
and writes `traffic_dataset.csv`.

### `core/graph_model.py`

Road-graph utilities: OSM network loading, nearest-node lookup, subgraphs,
shortest-path matrices, parallel-edge selection, path metrics, edge objective
weights, and distance calculation.

### `core/heuristics.py`

Greedy constructive routing using Dijkstra or A*. It evaluates candidate stops
with capacity, travel, waiting, service, and lateness considerations, then repairs
partial greedy results with a complete feasible order.

### `core/qpso.py`

The isolated quantum-inspired swarm optimizer. It knows nothing about maps,
vehicles, or customers; it only calls an evaluation function supplied by the
engine and searches continuous random-key vectors.

### `core/stop_selection.py`

Chooses connected, coordinate-backed, geographically spread depot/customer nodes
using deterministic seeded farthest-point sampling.

### `core/traffic.py`

Loads the bundled traffic CSV, normalizes traffic-mode aliases, applies matching
edge traffic records, and generates deterministic fallback traffic for unmatched
edges.

### `core/visualization.py`

Streamlit/Matplotlib visualizations: physical route map, abstract graph view, and
Gantt-style time view. This is separate from the browser MapLibre frontend.

### `core/vrp.py`

Core data structures and route mathematics:

- `Customer`;
- `VRPInstance`;
- capacity feasibility;
- feasible order construction;
- random-key route decoding;
- route objective evaluation;
- 2-opt route improvement.

### `core/traffic_dataset.csv`

Bundled edge-level off-peak/rush-hour traffic records. It is data rather than
Python code, but it is required for deterministic traffic matching.

## 19. Other important project files

### `app.py`

Optional Streamlit application. It loads a graph, applies traffic, creates an
instance, runs A*/QPSO/GA, and renders analysis charts using `core.visualization`.

### `requirements.txt`

Python runtime dependencies and compatible version ranges.

### `vrp-dashboard/package.json`

Frontend runtime and development dependencies. Install them with `npm ci` from
inside `vrp-dashboard`.

### `documentation/run.md`

Local installation, environment variables, startup commands, validation, and
troubleshooting instructions.

### `tests/`

Regression and benchmark tests. They cover evaluator behavior, time windows,
capacity failure handling, customer-set consistency, and benchmark scenarios where
QPSO is expected to perform strongly.

## 20. How to interpret results correctly

### Objective score is not money

The `score` is the mathematical optimization objective. It combines seconds,
meters multiplied by a configurable weight, service time, waiting time, and a
large lateness penalty. It is not a rupee amount.

The business-impact card separately estimates fuel, rupee, distance, and CO2
savings using fixed reporting assumptions.

### Distance and travel time are not the complete score

One route can have lower distance and lower driving time but a higher objective if
it causes more early-arrival waiting or service-window lateness. This is expected
under the current objective.

### Compare algorithms fairly

For a fair comparison, keep identical:

- graph and traffic mode;
- customer nodes and demands;
- time windows;
- fleet size and capacity;
- distance weight;
- random seed where relevant;
- population/swarm budget.

Run multiple seeds before making a scientific claim about superiority.

## 21. Debugging workflow

### Backend errors

Read the FastAPI terminal first. Common response classes are:

- `400`: graph loading, stop selection, or fleet feasibility input problem;
- `422`: an algorithm produced non-finite metrics, an unreachable leg, or an
  incomplete/duplicate customer set.

Useful checks:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q core app.py
```

For a specific instance, call the API through `http://localhost:8000/docs` and
inspect the JSON response before debugging the map.

### Frontend errors

Check:

```powershell
cd vrp-dashboard
npm run lint
npx tsc --noEmit
```

Then confirm:

- `NEXT_PUBLIC_API_URL` points to the backend;
- `NEXT_PUBLIC_STADIA_API_KEY` is set;
- the browser Network panel contains a successful `/api/optimize` response;
- the selected algorithm key exists in `routes_by_algorithm`;
- route features contain at least two coordinates.

### Map appears without route colors

Separate the problem into layers:

1. Does the API response contain `routes_by_algorithm` features?
2. Do features contain valid longitude/latitude coordinates?
3. Does the selected algorithm have the same customer set?
4. Is MapLibre loaded and are its sources present?
5. Is the SVG fallback receiving projected route data?
6. Is the route panel’s vehicle filter hiding all routes?

The map provider supplies the background tiles; route colors are generated by the
application. A Stadia tile issue should not change the GeoJSON route data.

## 22. How to improve the project safely

### Change the objective

Modify `evaluate_routes` in `core/vrp.py`, then update:

- algorithm comparisons;
- dashboard metric labels;
- tests;
- documentation.

Keep all algorithms on the same evaluator if benchmarking fairness matters.

### Improve QPSO

Potential extensions include:

- adaptive `alpha` schedules;
- discrete permutation-aware perturbations;
- route-split optimization rather than only sequential decoding;
- penalty repair instead of infinite infeasibility scores;
- multiple independent swarms;
- local search using time-window-aware moves;
- archive/diversity preservation;
- multiobjective or Pareto QPSO for time, distance, and emissions.

When changing QPSO, test both continuous-position validity and decoded route
validity.

### Improve GA

Possible extensions include:

- route-aware crossover;
- insertion and inversion mutations;
- adaptive mutation rates;
- local search after crossover;
- explicit capacity-preserving chromosome repair;
- multiple populations or island models.

### Improve A*

The current A* algorithm optimizes individual road legs and greedily chooses the
next customer. Future work could use:

- look-ahead customer scoring;
- regret insertion;
- time-window feasibility pruning;
- beam search;
- route-level A* for small instances;
- an exact/label-setting method for constrained subproblems.

### Improve traffic realism

The current system is scenario-based and deterministic, not a live GPS traffic
system. Future integrations could add:

- a live provider adapter;
- time-dependent edge weights;
- vehicle-dependent travel times;
- historical calibration;
- incident and road-closure layers;
- stochastic repeated-scenario evaluation.

### Improve scientific benchmarking

Use a benchmark harness that records for every seed:

- score;
- driving time;
- distance;
- waiting time;
- lateness;
- service time;
- runtime;
- feasibility;
- number of vehicles used.

Report mean, standard deviation, best, worst, and confidence intervals across
multiple seeds. A single dashboard run is useful for demonstration, not proof of
algorithmic dominance.

## 23. Extension checklist for collaborators

Before opening a change:

1. Identify whether the change affects graph data, traffic, decoding, scoring,
   optimization, API serialization, or UI rendering.
2. Preserve the same customer instance when comparing algorithms.
3. Add or update a regression test for the changed behavior.
4. Run Python tests and compilation checks.
5. Run TypeScript and ESLint checks for frontend changes.
6. Confirm that invalid or disconnected data fails with a useful error.
7. Update `run.md`, this guide, or parameter documentation if user behavior changes.
8. Check that route animation and `routes_by_algorithm` still agree.

## 24. Final mental model

The simplest accurate way to remember the project is:

```text
OSM tells us where roads are.
Traffic tells us how expensive each road is right now.
The stop matrix tells us the best road path between selected stops.
The decoder turns an ordering into vehicle routes.
The evaluator measures time, distance, waiting, service, and lateness.
QPSO searches many orderings using continuous random keys.
GA searches permutations using evolutionary operations.
A* greedily builds a route while using A* for each road leg.
FastAPI packages the results.
The Next.js dashboard animates and compares them.
```

That separation of concerns is the main design feature of the project. It lets
future collaborators improve the traffic model, route decoder, objective,
optimizer, or interface independently while keeping the overall workflow
understandable and testable.
