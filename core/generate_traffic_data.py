import os

import osmnx as ox
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import beta

def generate_bpr_traffic_model(place_name="Salt Lake, Kolkata, India", output_file="traffic_dataset.csv", seed=42):
    print(f"Ingesting GIS Network for {place_name}...")
    G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
    
    rng = np.random.default_rng(seed)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    
    # 1. Establish Free-Flow Speeds based on Road Hierarchy
    speed_limits = {
        'motorway': 60.0, 'trunk': 50.0, 'primary': 40.0, 
        'secondary': 35.0, 'tertiary': 30.0, 'residential': 20.0
    }
    
    def get_base_speed(hw_type):
        if isinstance(hw_type, list): hw_type = hw_type[0]
        return speed_limits.get(hw_type, 25.0)

    edges['free_flow_kph'] = edges['highway'].apply(get_base_speed)
    
    # Extract node coordinates for spatial bottleneck modeling
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    node_coords = np.array([[data['x'], data['y']] for _, data in nodes.iterrows()])
    node_ids = nodes.index.to_numpy()
    tree = cKDTree(node_coords)
    
    # 2. Generate Congestion Hotspots (e.g., Major Intersections)
    # 5% of the network acts as severe traffic sinks
    num_hotspots = max(1, int(len(node_coords) * 0.05))
    hotspot_indices = rng.choice(len(node_coords), size=num_hotspots, replace=False)
    hotspot_coords = node_coords[hotspot_indices]
    
    edge_records = []
    
    print("Applying BPR Link Performance Function & Beta Distributions...")
    for u, v, key , data in G.edges(keys=True, data=True):
        length_m = float(data.get('length', 10.0))
        free_flow_kph = get_base_speed(data.get('highway', 'unclassified'))
        free_flow_time_s = length_m / (free_flow_kph * 0.27778)
        
        # Calculate spatial proximity to nearest hotspot (Gaussian decay)
        edge_x = G.nodes[u]['x']
        edge_y = G.nodes[u]['y']
        dist_to_hotspot, _ = tree.query([edge_x, edge_y], k=1)
        spatial_penalty = np.exp(-(dist_to_hotspot**2) / 0.0001) # Decay parameter
        
        # 3. Model V/C (Volume/Capacity) Ratios using Beta Distribution
        # Off-Peak: Most roads are empty (Beta 2,5), small variance
        vc_offpeak = beta.rvs(a=2, b=5, random_state=rng)
        
        # Rush Hour: High saturation (Beta 6,2) + Hotspot Proximity
        vc_rush = beta.rvs(a=6, b=2, random_state=rng) + (spatial_penalty * 0.5)
        vc_rush = min(vc_rush, 1.5) # Cap at 150% capacity (gridlock)
        
        # 4. Apply standard BPR Formula: T = T_f * (1 + 0.15 * (V/C)^4)
        alpha, beta_param = 0.15, 4.0
        
        time_offpeak_s = free_flow_time_s * (1 + alpha * (vc_offpeak ** beta_param))
        time_rush_s = free_flow_time_s * (1 + alpha * (vc_rush ** beta_param))
        
        # Convert back to actual observed kinematic speeds
        speed_offpeak_kph = (length_m / time_offpeak_s) * 3.6
        speed_rush_kph = (length_m / time_rush_s) * 3.6
        
        edge_records.append({
            'u': u,
            'v': v,
            'key': key,
            'length_m': round(length_m, 2),
            'free_flow_kph': round(free_flow_kph, 1),
            'speed_offpeak_kph': round(speed_offpeak_kph, 2),
            'speed_rush_kph': round(speed_rush_kph, 2),
            'travel_time_offpeak_s': round(time_offpeak_s, 2),
            'travel_time_rush_s': round(time_rush_s, 2)
        })
        
    df = pd.DataFrame(edge_records)
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), output_file)
    df.to_csv(output_file, index=False)
    print(f"Statistically accurate dataset saved to {output_file}. Edges processed: {len(df)}")

if __name__ == "__main__":
    generate_bpr_traffic_model()
