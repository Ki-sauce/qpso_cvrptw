# Quantum-Inspired QPSO for CVRPTW

This project is a real-road-network delivery route optimizer for the
**Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**.

It compares three routing strategies on the same depot, customers, demands,
fleet, traffic scenario, and time windows:

- **QPSO** — Quantum-inspired Particle Swarm Optimization;
- **GA** — permutation-aware Genetic Algorithm;
- **A*** — greedy constructive routing with A* road-leg search.

The main interface is a Next.js/MapLibre dashboard backed by a FastAPI Python
service. An optional Streamlit interface is also included for analysis.

## What the project solves

Given:

- a real OpenStreetMap drivable road network;
- a depot and delivery locations;
- customer demands;
- a fleet size and vehicle capacity;
- customer ready/due time windows;
- service/unloading durations;
- traffic-adjusted road travel times;

the system searches for vehicle routes that visit every customer exactly once,
respect vehicle capacity, return vehicles to the depot, and minimize a shared
operational objective.

Real-world analogy: a dispatch manager is assigning delivery stops to vans,
choosing the visiting order, routing each road leg, and avoiding unnecessary
driving, waiting, and late deliveries.

## End-to-end workflow

```text
User configures the dashboard
              |
              v
Next.js sends POST /api/optimize
              |
              v
FastAPI validates the request
              |
              v
OSMnx loads/caches the drivable road graph
              |
              v
Traffic scenario updates edge travel times
              |
              v
Connected, spatially spread stops are selected
              |
              v
Demands and time windows are generated deterministically
              |
              +-------------+-------------+
              |             |             |
              v             v             v
             A*            GA           QPSO
              |             |             |
              +-------------+-------------+
                            v
                 Shared route evaluator
                            |
                            v
              Validated GeoJSON and metrics
                            |
                            v
              Map animation, schedules, benchmarks
```

## Architecture

```text
Browser :3000
   |
   +--> Next.js dashboard
   |       - controls and benchmark table
   |       - MapLibre/Stadia map
   |       - route animation and node colors
   |       - schedule and benchmark views
   |
   +--> FastAPI :8000
           - OSM graph preparation
           - traffic application
           - A*/GA/QPSO execution
           - validation and serialization
```

The optional Streamlit application runs `app.py` directly and uses the same
modules inside `core` without requiring the FastAPI frontend workflow.

## Objective function

For a candidate solution, the evaluator in `core/vrp.py` computes:

```text
objective =
    time_weight * (travel_time + waiting_time + service_time)
  + distance_weight * distance
  + lateness_penalty
```

The current lateness penalty is:

```text
lateness_penalty = (total_lateness_seconds / 60) * 10000
```

This is an optimization score, not a monetary value. It combines seconds and a
weighted distance term. The dashboard separately estimates fuel, rupee, distance,
and CO2 impact.

For each customer:

```text
arrival       = current_time + travel_time
service_start = max(arrival, ready_time)
waiting       = max(0, ready_time - arrival)
lateness      = max(0, service_start - due_time)
```

Early arrival makes the vehicle wait. Late arrival creates an SLA penalty.

## Road graph and traffic

OSMnx downloads a drivable OpenStreetMap graph. The backend keeps the largest
connected or strongly connected component, patches missing edge distances and
travel times, and caches the graph during the process.

Each edge receives:

```text
_routing_weight = time_weight * travel_time_s
                + distance_weight * distance_m
```

NetworkX shortest paths are precomputed between the depot and selected customers.
The optimizer works with this stop-to-stop matrix rather than repeatedly solving
the full road graph.

Traffic modes currently map to deterministic scenarios:

```text
simulated -> off_peak
live      -> rush_hour/mock
```

Matching edges use `core/traffic_dataset.csv`. Unmatched edges use a deterministic
fallback travel-time multiplier. The `live` label currently does not call a live
external traffic API.

`core/generate_traffic_data.py` can generate a new BPR-style traffic dataset using:

```text
T = T_free_flow * (1 + 0.15 * (V/C)^4)
```

## Route representation

QPSO searches continuous random-key vectors instead of directly searching binary
road-edge variables.

Example:

```text
particle = [0.71, 0.13, 0.94, 0.31]
```

Sorting the values produces a customer order:

```text
C2 -> C4 -> C1 -> C3
```

`split_random_key_solution` then converts that order into vehicle routes while
respecting capacity:

```text
[Depot, C2, C4, Depot]
[Depot, C1, C3, Depot]
```

The same route decoder and evaluator are used by QPSO, GA, and the shared
benchmarking engine, which makes their scores comparable.

## QPSO

QPSO operates on a swarm of vectors in `[0, 1]^N`.

For each iteration it computes the mean personal-best position:

```text
mbest = mean(pbest positions)
```

It then creates an attractor between the particle’s personal best and global best:

```text
p = phi * pbest + (1 - phi) * gbest
```

The position update is:

```text
X_new = p +/- alpha * |mbest - X| * ln(1/u)
```

where `phi` and `u` are random values in `(0,1)`. The project decreases `alpha`
from `1.0` to `0.5`, moving from broad exploration toward exploitation.

QPSO also applies a 2-opt local route improvement after decoding. The initial
swarm contains one capacity-feasible seeded solution, while other particles
explore random orderings.

QPSO is a metaheuristic. It seeks high-quality near-optimal solutions, but it does
not prove a global optimum for large NP-hard instances. The exact solver is only
intended for small instances.

## GA

The Genetic Algorithm works directly with integer customer permutations.

Each generation performs:

1. route evaluation;
2. three-way tournament selection;
3. order crossover;
4. optional swap mutation;
5. elitism, preserving the best solution.

The resulting permutation is passed through the same capacity decoder and
objective evaluator used by QPSO.

## A* baseline

A* is used to find a good road path for each individual leg. The customer order is
constructed greedily by selecting the locally best feasible next customer based on
travel cost, waiting, service time, and lateness.

This makes A* a useful practical baseline, but it is not a global VRP optimizer.
It can make a locally attractive choice that harms the remainder of the route. If
greedy bin packing leaves customers unserved, the implementation repairs the
order with a complete capacity-feasible assignment before returning the result.

## Fair comparison guarantees

The API runs all algorithms on the same:

- road graph;
- traffic-adjusted edge times;
- depot;
- customer node set;
- demands;
- time windows;
- fleet capacity;
- vehicle count;
- distance weight.

The API rejects results that contain:

- non-finite metrics;
- unreachable route legs;
- missing customers;
- duplicate customer visits.

This prevents the dashboard from comparing a complete QPSO solution with a
partial GA or A* solution.

## Main project files

### Backend and algorithms

| File | Responsibility |
| --- | --- |
| `core/api.py` | FastAPI endpoint, request validation, graph orchestration, algorithm execution, response serialization |
| `core/engine.py` | Shared optimizer construction, random-key decoding, evaluation adapter, result payloads |
| `core/qpso.py` | Quantum-inspired swarm update and convergence tracking |
| `core/ga.py` | Permutation GA, crossover, mutation, tournament selection, elitism |
| `core/heuristics.py` | Greedy Dijkstra/A* construction, time-window scoring, completeness repair |
| `core/baselines.py` | Random search, exact small-instance enumeration, gap calculation |
| `core/vrp.py` | Customer/instance models, capacity logic, route decoding, objective evaluation, 2-opt |
| `core/graph_model.py` | OSM graph utilities, shortest-path matrix, edge metrics, routing weights |
| `core/traffic.py` | Traffic CSV loading, traffic-mode normalization, edge-time updates |
| `core/stop_selection.py` | Connected and spatially spread stop selection |
| `core/generate_traffic_data.py` | BPR-style traffic dataset generation |
| `core/visualization.py` | Streamlit/Matplotlib route, graph, and Gantt visualizations |
| `core/traffic_dataset.csv` | Bundled edge-level traffic data |

### Applications and frontend

| File or folder | Responsibility |
| --- | --- |
| `app.py` | Optional Streamlit application |
| `vrp-dashboard/app/page.tsx` | Main dashboard state, controls, API calls, benchmark views |
| `vrp-dashboard/components/MapViewport.tsx` | MapLibre/Stadia map, routes, nodes, animation, playback controls |
| `vrp-dashboard/components/TimelineGantt.tsx` | Schedule and time-window visualization |
| `vrp-dashboard/components/DraggablePanel.tsx` | Draggable/resizable floating panels |
| `vrp-dashboard/app/globals.css` | Dark UI theme, map controls, responsive layout |
| `requirements.txt` | Python dependencies |
| `vrp-dashboard/package.json` | JavaScript dependencies and frontend scripts |

## Install and run

Read the complete setup instructions in [documentation/run.md](documentation/run.md).

Short version:

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn core.api:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

In another terminal:

```powershell
cd vrp-dashboard
npm ci
npm run dev
```

Create `vrp-dashboard/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_STADIA_API_KEY=your_stadia_maps_key
```

Then open [http://localhost:3000](http://localhost:3000).

### Optional Streamlit interface

```powershell
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Validation commands

From the project root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q core app.py
```

From `vrp-dashboard`:

```powershell
npm run lint
npx tsc --noEmit
```

## Important limitations

- QPSO is not guaranteed to find a global optimum for large NP-hard instances.
- `solve_exact` is practical only for small customer counts.
- Large-instance capacity feasibility uses a bounded approximation after 24 customers.
- The current `live` traffic mode is a rush-hour/mock scenario, not a live provider.
- Public OSM downloads require internet access and may take time or be rate-limited.
- The objective score is not a currency amount.
- Multiple seeds should be used before claiming one algorithm is universally better.

## Documentation map

- [Run instructions](documentation/run.md) — installation, startup, environment variables, and troubleshooting.
- [Project technical guide](documentation/project_guide.md) — full workflow, mathematics, algorithms, file explanations, and extension guidance.
- [Parameter reference](documentation/Parameters.md) — environment and optimizer parameters.
- [Randomness notes](documentation/Random.md) — seeded behavior and stochastic components.

## Final mental model

```text
OpenStreetMap supplies the roads.
Traffic supplies edge travel times.
Stop selection supplies a connected delivery instance.
The decoder turns customer orderings into vehicle routes.
The evaluator measures time, distance, waiting, service, and lateness.
QPSO searches continuous random-key orderings.
GA searches permutations through evolution.
A* greedily builds a route using shortest road legs.
FastAPI packages all results.
The Next.js dashboard visualizes and animates them.
```
