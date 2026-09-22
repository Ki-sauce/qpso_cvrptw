import unittest

import networkx as nx

from core.engine import build_optimizer, solve_ga_baseline, solve_qpso
from core.graph_model import route_distance
from core.heuristics import solve_dynamic_heuristic
from core.vrp import Customer, VRPInstance


def make_graph(graph_type=nx.MultiDiGraph):
    graph = graph_type()
    for node in (0, 1, 2):
        graph.add_node(node, x=float(node), y=0.0)
    for u, v in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
        graph.add_edge(u, v, travel_time_s=100.0, distance_m=100.0)
    return graph


def make_complete_graph(node_count):
    graph = nx.DiGraph()
    for node in range(node_count):
        graph.add_node(node, x=float(node), y=0.0)
    for source in range(node_count):
        for target in range(node_count):
            if source != target:
                graph.add_edge(source, target, travel_time_s=1.0, distance_m=1.0)
    return graph


class CoreRegressionTests(unittest.TestCase):
    def test_time_window_uses_travel_time(self):
        instance = VRPInstance(
            depot=0,
            customers=[
                Customer(1, 1, ready_time=32400, due_time=32450, service_time=300),
                Customer(2, 1, ready_time=32400, due_time=40000, service_time=300),
            ],
            vehicle_capacity=10,
            num_vehicles=1,
        )
        evaluate, _ = build_optimizer(make_graph(), instance)
        _, metrics = evaluate([0.0, 1.0])
        self.assertGreater(metrics["tw_penalty"], 0)
        self.assertGreater(metrics["total_lateness_s"], 0)

    def test_time_weight_changes_score(self):
        instance = VRPInstance(0, [Customer(1, 1)], 10, 1)
        graph = make_graph()
        score_one = build_optimizer(graph, instance, time_weight=1.0)[0]([0.5])[0]
        score_two = build_optimizer(graph, instance, time_weight=2.0)[0]([0.5])[0]
        self.assertNotEqual(score_one, score_two)

    def test_oversized_customer_is_infeasible_not_a_crash(self):
        instance = VRPInstance(0, [Customer(1, 7)], 5, 1)
        evaluate, _ = build_optimizer(make_graph(), instance)
        score, _ = evaluate([0.5])
        self.assertEqual(score, float("inf"))

    def test_simple_graph_distance_is_supported(self):
        graph = make_graph(nx.DiGraph)
        self.assertEqual(route_distance(graph, [0, 1]), 100.0)

    def test_all_algorithms_serve_the_same_customer_set(self):
        # The greedy A* choice 1 -> 2 can leave customer 4 unserved when the
        # demands are (2, 2, 7, 7).  The instance is nevertheless feasible as
        # [7, 2] + [7, 2], so the constructive solver must repair its order.
        graph = make_complete_graph(5)
        instance = VRPInstance(
            depot=0,
            customers=[
                Customer(1, 2, ready_time=0, due_time=999999, service_time=0),
                Customer(2, 2, ready_time=0, due_time=999999, service_time=0),
                Customer(3, 7, ready_time=0, due_time=999999, service_time=0),
                Customer(4, 7, ready_time=0, due_time=999999, service_time=0),
            ],
            vehicle_capacity=10,
            num_vehicles=2,
        )
        expected = {1, 2, 3, 4}

        solutions = [
            solve_dynamic_heuristic(graph, instance, method="astar"),
            solve_ga_baseline(graph, instance, particles=8, iterations=8, seed=7),
            solve_qpso(graph, instance, particles=8, iterations=8, seed=7),
        ]
        for solution in solutions:
            served = [
                node
                for route in solution["routes"]
                for node in route
                if node != instance.depot
            ]
            self.assertEqual(set(served), expected)
            self.assertEqual(len(served), len(expected))


if __name__ == "__main__":
    unittest.main()
