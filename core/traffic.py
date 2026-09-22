import os

import networkx as nx
import numpy as np
import pandas as pd


TRAFFIC_DATA_MAP = None


def load_traffic_data(csv_path=None):
    global TRAFFIC_DATA_MAP
    if TRAFFIC_DATA_MAP is not None and csv_path is None:
        return TRAFFIC_DATA_MAP

    csv_path = csv_path or os.path.join(os.path.dirname(__file__), "traffic_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Traffic dataset not found at {csv_path}. Run generate_traffic_data.py first."
        )

    df = pd.read_csv(csv_path)
    pair_map = {}
    keyed_map = {}
    has_key = "key" in df.columns
    for _, row in df.iterrows():
        record = {
            "travel_time_offpeak_s": float(row["travel_time_offpeak_s"]),
            "travel_time_rush_s": float(row["travel_time_rush_s"]),
            "speed_offpeak_kph": float(row["speed_offpeak_kph"]),
            "speed_rush_kph": float(row["speed_rush_kph"]),
        }
        u, v = int(row["u"]), int(row["v"])
        pair_map.setdefault((u, v), []).append(record)
        if has_key and not pd.isna(row["key"]):
            keyed_map[(u, v, int(row["key"]))] = record

    TRAFFIC_DATA_MAP = {"pairs": pair_map, "edges": keyed_map}
    return TRAFFIC_DATA_MAP


def _normalise_mode(mode):
    aliases = {
        "simulated": "off_peak",
        "offpeak": "off_peak",
        "off_peak": "off_peak",
        "live": "rush_hour",
        "rush": "rush_hour",
        "rush_hour": "rush_hour",
    }
    try:
        return aliases[mode.lower()]
    except (AttributeError, KeyError):
        raise ValueError("traffic mode must be simulated/off_peak or live/rush_hour")


def apply_traffic_scenario(G, mode="off_peak", seed=None, csv_path=None):
    """Apply deterministic traffic data to a graph.

    ``seed`` remains accepted for compatibility with the former stochastic
    implementation. The current CSV-backed scenarios are deterministic.
    """
    rng = np.random.default_rng(42 if seed is None else seed)
    mode = _normalise_mode(mode)
    traffic_data = load_traffic_data(csv_path)
    G_live = G.copy()
    pair_map = traffic_data["pairs"]
    keyed_map = traffic_data["edges"]
    is_multi = isinstance(G_live, (nx.MultiGraph, nx.MultiDiGraph))
    edge_iter = (
        G_live.edges(keys=True, data=True)
        if is_multi
        else G_live.edges(data=True)
    )

    for edge in edge_iter:
        if is_multi:
            u, v, key, data = edge
            traffic_attrs = keyed_map.get((u, v, key))
        else:
            u, v, data = edge
            traffic_attrs = None

        candidates = pair_map.get((u, v), [])
        if traffic_attrs is None and candidates:
            time_key = (
                "travel_time_rush_s"
                if mode == "rush_hour"
                else "travel_time_offpeak_s"
            )
            traffic_attrs = min(candidates, key=lambda item: item[time_key])

        length_m = float(data.get("length", data.get("distance_m", 10.0)))
        data["distance_m"] = length_m

        if traffic_attrs is not None:
            if mode == "rush_hour":
                data["travel_time_s"] = traffic_attrs["travel_time_rush_s"]
                data["speed_kmh"] = traffic_attrs["speed_rush_kph"]
            else:
                data["travel_time_s"] = traffic_attrs["travel_time_offpeak_s"]
                data["speed_kmh"] = traffic_attrs["speed_offpeak_kph"]
        else:
            existing_time = float(data.get("travel_time_s", 0.0))
            if existing_time <= 0 or not pd.notna(existing_time):
                speed_ms = 25.0 * (5.0 / 18.0)
                existing_time = length_m / speed_ms
            # For locations not covered by the bundled dataset, retain the
            # graph's baseline but still make the requested traffic scenario
            # deterministic and seed-controlled.
            multiplier = rng.normal(
                loc=1.35 if mode == "rush_hour" else 1.10,
                scale=0.10 if mode == "rush_hour" else 0.05,
            )
            multiplier = max(1.0, float(multiplier))
            data["travel_time_s"] = existing_time * multiplier
            data["speed_kmh"] = length_m / data["travel_time_s"] * 3.6

    return G_live
