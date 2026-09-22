# core/ml_warmstart.py
import torch
import torch.nn as nn
import numpy as np
import os
from core.device import get_device

class TransformerKeyPredictor(nn.Module):
    def __init__(self, feat_dim=8, d_model=128, n_heads=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)

_MODEL_CACHE = None

def get_model(weights_path="models/transformer_warmstart.pt"):
    global _MODEL_CACHE
    device = get_device()
    if _MODEL_CACHE is None:
        model = TransformerKeyPredictor().to(device)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded warm-start model weights from {weights_path}")
        else:
            print(f"Warning: Weights not found at {weights_path}. Model uninitialized!")
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE

def extract_features(instance, G):
    """Extracts the 8-dim feature matrix using VRPInstance and OSM Graph nodes."""
    depot_node = instance.depot
    depot_coords = np.array([G.nodes[depot_node]['x'], G.nodes[depot_node]['y']], dtype=np.float32)
    
    cust_coords = np.array([[G.nodes[c.node]['x'], G.nodes[c.node]['y']] for c in instance.customers], dtype=np.float32)
    demands = np.array([c.demand for c in instance.customers], dtype=np.float32)
    tw_starts = np.array([c.ready_time for c in instance.customers], dtype=np.float32)
    tw_ends = np.array([c.due_time for c in instance.customers], dtype=np.float32)
    
    rel_x = (cust_coords[:, 0] - depot_coords[0]) / 100.0
    rel_y = (cust_coords[:, 1] - depot_coords[1]) / 100.0
    depot_dist = np.linalg.norm(cust_coords - depot_coords, axis=-1) / 100.0
    
    max_horizon = tw_ends.max() if len(tw_ends) > 0 else 86400.0

    return np.stack([
        cust_coords[:, 0] / 100.0,
        cust_coords[:, 1] / 100.0,
        rel_x,
        rel_y,
        demands / instance.vehicle_capacity,
        tw_starts / max_horizon,
        tw_ends / max_horizon,
        depot_dist
    ], axis=-1).astype(np.float32)

def generate_warm_keys(instance, G, n_particles=30, warm_start_fraction=0.5):
    device = get_device()
    model = get_model()
    features = extract_features(instance, G)
    
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        base_key = model(x_tensor).squeeze(0).cpu().numpy()

    n_warm = int(n_particles * warm_start_fraction)
    noise = np.random.normal(0, 0.02, size=(n_warm, len(base_key)))
    warm_keys = np.clip(base_key[None, :] + noise, 0, 1).astype(np.float32)
    
    return warm_keys