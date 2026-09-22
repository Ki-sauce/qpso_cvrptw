import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

def generate_node_labels(instance):
    """Creates a universal mapping of Node IDs to clean labels (e.g., 'C1', 'Depot')."""
    labels = {instance.depot: "Depot"}
    for i, c in enumerate(instance.customers):
        labels[c.node] = f"C{i+1}"
    return labels

def edge_geometry(G, u, v):
    """Safely extracts geometry for plotting spatial routes."""
    data = G.get_edge_data(u, v)
    if data is None: return None
    if G.is_multigraph():
        key, attrs = min(
            list(data.items()),
            key=lambda item: item[1].get("_routing_weight", item[1].get("travel_time_s", float("inf"))),
        )
    else:
        attrs = data
    geometry = attrs.get("geometry")
    if geometry is not None: return np.asarray(geometry.coords)
    return np.asarray([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]])

def plot_map_view(G, result, instance):
    """🗺️ Spatial View: Scales dynamically to prevent clutter at high N."""
    labels = generate_node_labels(instance)
    N = len(instance.customers)
    
    # Dynamic scaling based on customer count
    line_w = max(1.0, 2.5 - (N * 0.05))
    node_s = max(15, 60 - N)
    font_s = max(5, 9 - (N // 10))
    
    fig, ax = ox.plot_graph(
        G, show=False, close=False, node_size=0,
        edge_color="#eef0f2", edge_linewidth=0.35, bgcolor="white", figsize=(10, 5)
    )
    ax.set_axis_off()
    
    route_colors = ["#D90429", "#0077B6", "#2A9D8F", "#F4A261", "#7209B7", "#FFB703", "#FB8500"]
    legend_handles = []

    # 1. Plot physical routes
    for vehicle_idx, route in enumerate(result["routes"]):
        if len(route) <= 2 and route[0] == route[-1]: continue
        vehicle_segments = []
        color = route_colors[vehicle_idx % len(route_colors)]

        for a, b in zip(route[:-1], route[1:]):
            path = result.get("paths", {}).get((a, b))
            if not path or len(path) < 2: continue
            for u, v in zip(path[:-1], path[1:]):
                coords = edge_geometry(G, u, v)
                if coords is not None and len(coords) >= 2:
                    vehicle_segments.append(coords)

        if vehicle_segments:
            lc = LineCollection(vehicle_segments, colors=[color], linewidths=line_w, alpha=0.85, zorder=4, capstyle="round")
            ax.add_collection(lc)
            legend_handles.append(Line2D([0], [0], color=color, lw=2, label=f"Vehicle {vehicle_idx + 1}"))

    # 2. Plot Nodes with scaled labels
    for c in instance.customers:
        if c.node in G.nodes:
            x, y = G.nodes[c.node]["x"], G.nodes[c.node]["y"]
            ax.scatter(x, y, s=node_s, marker="o", facecolor="white", edgecolor="#111827", linewidth=1.0, zorder=7)
            
            # Hide labels entirely if N is massive (>40), otherwise scale them
            if N <= 40:
                ax.annotate(labels[c.node], (x, y), xytext=(4, 4), textcoords="offset points", 
                            fontsize=font_s, fontweight='bold', color="#111827", 
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6), zorder=9)

    if instance.depot in G.nodes:
        dx, dy = G.nodes[instance.depot]["x"], G.nodes[instance.depot]["y"]
        ax.scatter([dx], [dy], s=node_s*4, marker="*", facecolor="#D90429", edgecolor="white", linewidth=1.0, zorder=8)
        ax.annotate(labels[instance.depot], (dx, dy), xytext=(6, 6), textcoords="offset points", 
                    fontsize=font_s+1, fontweight='bold', color="#D90429",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=10)

    ax.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.9, edgecolor="#e5e7eb", fontsize=8, ncol=min(4, len(legend_handles)))
    fig.tight_layout(pad=0)
    return fig

import networkx as nx

def plot_graph_view(G, result, instance):
    """⏺️ Topological View: Uses force-directed physics to un-clutter the layout."""
    labels = generate_node_labels(instance)
    demands = {c.node: c.demand for c in instance.customers}
    N = len(instance.customers)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f9fafb")
    ax.axis("off")

    route_colors = ["#D90429", "#0077B6", "#2A9D8F", "#F4A261", "#7209B7", "#FFB703", "#FB8500"]
    legend_handles = []
    max_cap = instance.vehicle_capacity

    # 1. Build an abstract mathematical graph (ignore GPS entirely)
    H = nx.DiGraph()
    vehicle_routes = []
    
    for vehicle_idx, route in enumerate(result["routes"]):
        if len(route) <= 2 and route[0] == route[-1]: continue
        color = route_colors[vehicle_idx % len(route_colors)]
        route_load = sum(demands.get(n, 0) for n in route)
        
        # Add edges to the abstract graph
        for u, v in zip(route[:-1], route[1:]):
            H.add_edge(u, v, color=color)
            
        vehicle_routes.append((route, color))
        legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle="-", 
                                     label=f"Veh {vehicle_idx + 1} (Load: {route_load}/{max_cap})"))

    # 2. Apply Force-Directed Physics (Spring Layout)
    # k determines the optimal distance between nodes. Higher N = smaller k to fit on screen.
    pos = nx.spring_layout(H, k=2.0/np.sqrt(len(H.nodes)), seed=42, iterations=100)

    # 3. Draw abstract edges
    for u, v, data in H.edges(data=True):
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        ax.plot(x_coords, y_coords, color=data['color'], linewidth=1.5, alpha=0.7, zorder=4)
        
        # Add a clear directional arrow in the middle of the abstract edge
        mid_x, mid_y = (x_coords[0] + x_coords[1])/2, (y_coords[0] + y_coords[1])/2
        ax.annotate("", xy=(mid_x, mid_y), xytext=(x_coords[0], y_coords[0]), 
                    arrowprops=dict(arrowstyle="->", color=data['color'], lw=1.5, alpha=0.8))

    # 4. Draw abstract nodes
    base_font = max(6, 9 - (N // 15))
    
    for node in H.nodes:
        x, y = pos[node]
        if node == instance.depot:
            ax.scatter([x], [y], s=500, marker="*", facecolor="#D90429", edgecolor="white", linewidth=1.5, zorder=8)
            ax.text(x, y - 0.08, labels[node], fontsize=base_font+1, ha='center', fontweight='bold', color="#D90429")
        else:
            demand = demands.get(node, 0)
            bubble_size = max(100, 200 + (demand * 30)) # Scale by demand
            ax.scatter([x], [y], s=bubble_size, marker="o", facecolor="white", edgecolor="#111827", linewidth=1.5, zorder=7)
            ax.text(x, y, f"{labels[node]}\n({int(demand)})", fontsize=base_font, ha='center', va='center', fontweight='bold', color="#111827", zorder=8)

    ax.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.9, edgecolor="#e5e7eb", fontsize=8)
    fig.tight_layout(pad=0)
    return fig

def plot_gantt_chart(G, result, instance):
    """⏱️ Temporal View: Focuses strictly on time elapsed and service window compliance."""
    labels = generate_node_labels(instance)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_facecolor("#f9fafb")
    
    y_ticks = []
    y_labels = []
    route_colors = ["#D90429", "#0077B6", "#2A9D8F", "#F4A261", "#7209B7"]
    max_time = 0 
    
    for vehicle_idx, route in enumerate(result["routes"]):
        if len(route) <= 2 and route[0] == route[-1]: continue
            
        color = route_colors[vehicle_idx % len(route_colors)]
        y_pos = vehicle_idx * 10
        y_ticks.append(y_pos + 4)
        y_labels.append(f"Vehicle {vehicle_idx + 1}")
        
        current_time = 0.0
        clock_time = 32400.0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            path = result.get("paths", {}).get((u, v))
            
            segment_time = 0.0
            if path:
                for a, b in zip(path[:-1], path[1:]):
                    data = G.get_edge_data(a, b)
                    if G.is_multigraph():
                        attrs = min(data.values(), key=lambda x: x.get("travel_time_s", float('inf')))
                    else:
                        attrs = data
                    segment_time += attrs.get("travel_time_s", 0)
            
            # Transit Block
            ax.broken_barh([(current_time, segment_time)], (y_pos, 8), facecolors=color, alpha=0.7, edgecolor='white')
            current_time += segment_time
            clock_time += segment_time

            customer = next((c for c in instance.customers if c.node == v), None)
            if customer is not None and clock_time < customer.ready_time:
                waiting = customer.ready_time - clock_time
                ax.broken_barh([(current_time, waiting)], (y_pos, 8), facecolors="#9ca3af", alpha=0.45, hatch='..')
                current_time += waiting
                clock_time = customer.ready_time
            
            # Service Block (Assuming 10 mins / 600s unloading time)
            if customer is not None:
                ax.broken_barh([(current_time, 300)], (y_pos, 8), facecolors="#111827", hatch='///')
                # Inject Universal Label into the timeline block
                ax.text(current_time + 150, y_pos + 4, labels[v], color="white", fontsize=8, ha='center', va='center', fontweight='bold')
                current_time += 300
                clock_time += 300
                
        max_time = max(max_time, current_time)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontweight='bold', color="#4b5563")
    ax.set_xlabel("Time Elapsed (Seconds from Dispatch)", fontweight='bold')
    
    legend_elements = [
        Patch(facecolor='#e5e7eb', alpha=0.7, label='Transit / Driving Block'),
        Patch(facecolor='#111827', hatch='///', label='Service/Unload Block')
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig
