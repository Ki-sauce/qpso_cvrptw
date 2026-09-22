"""Reproducible benchmark cases where the project's QPSO is expected to win.

These are not claims that QPSO is universally optimal.  They are regression
cases for representative conditions where population-based search is useful:

* several competing customer orders and vehicle splits;
* a common nonlinear CVRPTW objective for all algorithms; and
* a small, fixed compute budget where the constructive A* baseline is greedy
  and GA has limited opportunity to improve its initial population.

All algorithms operate on the same graph, instance, objective weights, seed,
and population/iteration budget.  The synthetic graph is complete and
projected, so the test does not require OSM/network access.
"""

import math
import unittest

import networkx as nx
import numpy as np

from core.engine import solve_ga_baseline, solve_qpso
from core.heuristics import solve_dynamic_heuristic
from core.vrp import Customer, VRPInstance


TIME_WEIGHT = 1.0
DISTANCE_WEIGHT = 0.002
PARTICLES = 10
ITERATIONS = 10


def make_benchmark_case(seed=1, tight_windows=False, customers_n=8):
    """Build a deterministic, projected complete road graph and CVRPTW case."""
    rng = np.random.default_rng(seed)
    coordinates = [(0.0, 0.0)]
    coordinates.extend(
        (
            float(rng.uniform(-10_000.0, 10_000.0)),
            float(rng.uniform(-10_000.0, 10_000.0)),
        )
        for _ in range(customers_n)
    )

    graph = nx.MultiDiGraph()
    # Prevent the A* heuristic from interpreting these projected coordinates
    # as longitude/latitude values.
    graph.graph["crs"] = type("ProjectedCRS", (), {"is_geographic": False})()
    for node, (x, y) in enumerate(coordinates):
        graph.add_node(node, x=x, y=y)

    for source, (x1, y1) in enumerate(coordinates):
        for target, (x2, y2) in enumerate(coordinates):
            if source == target:
                continue
            distance = math.hypot(x2 - x1, y2 - y1)
            graph.add_edge(
                source,
                target,
                distance_m=distance,
                length=distance,
                # 10 m/s is a consistent free-flow speed for every edge.
                travel_time_s=distance / 10.0,
            )

    customers = []
    for node in range(1, customers_n + 1):
        if tight_windows:
            ready_time = 32_400.0 + float(rng.uniform(0.0, 6_000.0))
            due_time = ready_time + float(rng.uniform(1_800.0, 9_000.0))
        else:
            ready_time = 32_400.0
            due_time = 86_400.0
        customers.append(
            Customer(
                node=node,
                demand=float(rng.integers(1, 5)),
                ready_time=ready_time,
                due_time=due_time,
                service_time=300.0,
            )
        )

    return graph, VRPInstance(
        depot=0,
        customers=customers,
        vehicle_capacity=30.0,
        num_vehicles=4,
    )


def run_algorithms(graph, instance, seed):
    """Run every project algorithm against one identical benchmark case."""
    return {
        "A*": solve_dynamic_heuristic(
            graph,
            instance,
            method="astar",
            time_weight=TIME_WEIGHT,
            distance_weight=DISTANCE_WEIGHT,
        ),
        "GA": solve_ga_baseline(
            graph,
            instance,
            particles=PARTICLES,
            iterations=ITERATIONS,
            seed=seed,
            time_weight=TIME_WEIGHT,
            distance_weight=DISTANCE_WEIGHT,
        ),
        "QPSO": solve_qpso(
            graph,
            instance,
            particles=PARTICLES,
            iterations=ITERATIONS,
            seed=seed,
            time_weight=TIME_WEIGHT,
            distance_weight=DISTANCE_WEIGHT,
        ),
    }


def assert_valid_result(test_case, result):
    test_case.assertTrue(math.isfinite(float(result["score"])))
    test_case.assertTrue(result["routes"])
    test_case.assertEqual(
        sorted(
            customer
            for route in result["routes"]
            for customer in route
            if customer != 0
        ),
        list(range(1, 9)),
    )
    test_case.assertTrue(all(route[0] == 0 and route[-1] == 0 for route in result["routes"]))


class QPSOBenchmarkTests(unittest.TestCase):
    def test_qpso_beats_astar_and_ga_on_multimodal_routing_case(self):
        """QPSO finds a lower route objective than both baselines."""
        graph, instance = make_benchmark_case(seed=1)
        results = run_algorithms(graph, instance, seed=1)

        for result in results.values():
            assert_valid_result(self, result)

        self.assertLess(results["QPSO"]["score"], results["A*"]["score"])
        self.assertLess(results["QPSO"]["score"], results["GA"]["score"])

    def test_qpso_beats_astar_and_ga_with_tight_time_windows(self):
        """QPSO handles global ordering better than greedy construction here."""
        graph, instance = make_benchmark_case(seed=0, tight_windows=True)
        results = run_algorithms(graph, instance, seed=0)

        for result in results.values():
            assert_valid_result(self, result)

        self.assertLess(results["QPSO"]["score"], results["A*"]["score"])
        self.assertLess(results["QPSO"]["score"], results["GA"]["score"])


if __name__ == "__main__":
    unittest.main()
