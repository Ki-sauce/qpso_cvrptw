import osmnx as ox
import networkx as nx


def load_osm_network(place):
    """Download and project a drivable OSM street network."""
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    G = ox.project_graph(G)
    for _, _, data in G.edges(data=True):
        data.setdefault("distance_m", float(data.get("length", 10.0)))
        data.setdefault("travel_time_s", data["distance_m"] / 8.33)
    return G


def nearest_node(G, x, y):
    return ox.distance.nearest_nodes(G, X=x, Y=y)


def edge_weighted_subgraph(G, nodes):
    return G.subgraph(nodes).copy()


def build_stop_matrix(G, stops, weight="travel_time_s"):
    """
    Compute shortest-path cost and path between every pair of stops.
    Works with OSMnx MultiDiGraph/MultiGraph.
    """
    costs = {}
    paths = {}

    for source in stops:
        lengths, route_paths = nx.single_source_dijkstra(G, source, weight=weight)
        for target in stops:
            if target == source:
                costs[(source, target)] = 0.0
                paths[(source, target)] = [source]
            elif target in lengths:
                costs[(source, target)] = float(lengths[target])
                paths[(source, target)] = route_paths[target]
            else:
                costs[(source, target)] = float("inf")
                paths[(source, target)] = []

    return costs, paths


def edge_attributes(G, u, v, weight="travel_time_s"):
    """Return the attributes of the edge selected by a shortest-path result.

    NetworkX paths on a MultiDiGraph contain node IDs, not edge keys.  The
    path algorithm selects the parallel edge with the lowest requested weight,
    so all later metrics must select that same edge as well.
    """
    data = G.get_edge_data(u, v)
    if not data:
        return None
    if G.is_multigraph():
        return min(
            data.values(),
            key=lambda attrs: float(attrs.get(weight, float("inf"))),
        )
    return data


def path_metrics(G, path, weight="travel_time_s"):
    """Return travel time and physical distance for a node path."""
    total_time = 0.0
    total_distance = 0.0
    for u, v in zip(path[:-1], path[1:]):
        attrs = edge_attributes(G, u, v, weight=weight)
        if attrs is None:
            return float("inf"), float("inf")
        total_time += float(attrs.get("travel_time_s", 0.0))
        total_distance += float(attrs.get("distance_m", attrs.get("length", 0.0)))
    return total_time, total_distance


def add_objective_weights(G, time_weight=1.0, distance_weight=0.0):
    """Copy ``G`` and add a common scalar routing objective to every edge."""
    weighted = G.copy()
    for _, _, data in weighted.edges(data=True):
        travel_time = float(data.get("travel_time_s", float("inf")))
        distance = float(data.get("distance_m", data.get("length", 0.0)))
        data["_routing_weight"] = time_weight * travel_time + distance_weight * distance
    return weighted


def route_distance(G, path, weight="travel_time_s"):
    total = 0.0
    return path_metrics(G, path, weight=weight)[1]
