"""Deterministic, spatially spread delivery-stop selection."""

import math

import networkx as nx
import numpy as np


def choose_spread_stops(G, n, seed):
    """Select a connected depot and spatially distributed delivery stops.

    Uniform sampling by graph node count tends to cluster stops in dense
    neighbourhoods.  Farthest-point sampling keeps the selected stops spread
    across the usable map area while retaining deterministic seeded behavior.
    All candidates come from one connected component so every algorithm can
    route between the selected locations.
    """
    if n < 1:
        raise ValueError("At least one delivery stop is required.")

    components = (
        nx.strongly_connected_components(G)
        if G.is_directed()
        else nx.connected_components(G)
    )
    component = max(components, key=len, default=set())
    candidates = []
    coordinates = []
    for node in component:
        attributes = G.nodes[node]
        try:
            x = float(attributes["x"])
            y = float(attributes["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            candidates.append(node)
            coordinates.append((x, y))

    required = n + 1  # depot plus delivery stops
    if len(candidates) < required:
        raise ValueError(
            f"Network requires {required} connected, coordinate-backed nodes. "
            f"Only {len(candidates)} are available."
        )

    rng = np.random.default_rng(seed)

    # Avoid an O(n * |V|) scan on very large OSM extracts while retaining a
    # much larger candidate pool than the maximum dashboard stop count.
    max_candidates = max(12000, required)
    if len(candidates) > max_candidates:
        candidate_indices = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[index] for index in candidate_indices]
        coordinates = [coordinates[index] for index in candidate_indices]

    points = np.asarray(coordinates, dtype=float)
    latitude_scale = max(0.2, math.cos(math.radians(float(points[:, 1].mean()))))
    metric_points = points.copy()
    metric_points[:, 0] *= latitude_scale

    first = int(rng.integers(0, len(candidates)))
    selected_indices = [first]
    available = np.ones(len(candidates), dtype=bool)
    available[first] = False
    min_distances = np.sum((metric_points - metric_points[first]) ** 2, axis=1)
    min_distances[first] = -1.0

    while len(selected_indices) < required:
        available_indices = np.flatnonzero(available)
        distances = min_distances[available_indices]

        # Pick randomly among the most distant few candidates.  This keeps
        # seeded runs reproducible without making every seed choose identical
        # corner points of the service area.
        top_count = min(8, len(available_indices))
        top_positions = np.argpartition(distances, -top_count)[-top_count:]
        chosen_position = int(rng.choice(top_positions))
        chosen = int(available_indices[chosen_position])
        selected_indices.append(chosen)
        available[chosen] = False

        distance_to_chosen = np.sum((metric_points - metric_points[chosen]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distance_to_chosen)
        min_distances[~available] = -1.0

    selected_nodes = [candidates[index] for index in selected_indices]
    return selected_nodes[0], selected_nodes[1:]
