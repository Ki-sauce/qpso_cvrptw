import os
import torch
import torch.nn as nn
import numpy as np
from core.device import get_device

class EdgePredictorTransformer(nn.Module):
    def __init__(self, feat_dim=8, d_model=128, n_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.scale = np.sqrt(d_model)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        
        Q = self.W_q(h)
        K = self.W_k(h)
        
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        probs = torch.sigmoid(logits)
        
        eye = torch.eye(probs.size(1), device=probs.device).unsqueeze(0).bool()
        probs.masked_fill_(eye, 0.0)
        return probs

_MODEL_CACHE = None

def get_model(weights_path="models/edge_predictor.pt"):
    global _MODEL_CACHE
    device = get_device()
    if _MODEL_CACHE is None:
        model = EdgePredictorTransformer().to(device)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded Edge-Predicting Neural Oracle from {weights_path}")
        else:
            print(f"Warning: Neural weights not found at {weights_path}. Model uninitialized!")
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE

def extract_features(instance, G):
    depot_node = instance.depot
    depot_coords = np.array([G.nodes[depot_node]['x'], G.nodes[depot_node]['y']], dtype=np.float32)
    
    cust_coords = np.array([[G.nodes[c.node]['x'], G.nodes[c.node]['y']] for c in instance.customers], dtype=np.float32)
    demands = np.array([c.demand for c in instance.customers], dtype=np.float32)
    tw_starts = np.array([c.ready_time for c in instance.customers], dtype=np.float32)
    tw_ends = np.array([c.due_time for c in instance.customers], dtype=np.float32)
    
    rel_x = (cust_coords[:, 0] - depot_coords[0]) / 100.0
    rel_y = (cust_coords[:, 1] - depot_coords[1]) / 100.0
    depot_dist = np.linalg.norm(cust_coords - depot_coords, axis=-1) / 100.0
    
    return np.stack([
        cust_coords[:, 0] / 100.0,
        cust_coords[:, 1] / 100.0,
        rel_x,
        rel_y,
        demands / instance.vehicle_capacity,
        tw_starts / 86400.0,
        tw_ends / 86400.0,
        depot_dist
    ], axis=-1).astype(np.float32)

def predict_edge_probabilities(instance, G):
    """Returns an (N x N) edge probability matrix across all customers."""
    device = get_device()
    model = get_model()
    feat = extract_features(instance, G)
    x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        probs = model(x).squeeze(0).cpu().numpy()
    return probs

def generate_warm_keys(instance, G, n_particles=30, warm_start_fraction=0.5):
    """
    Decodes the N x N neural probability matrix into optimal topological 
    permutations, converting them into continuous random keys for QPSO.
    """
    n_cust = len(instance.customers)
    if n_cust < 2:
        return np.random.uniform(0, 1, size=(n_particles, n_cust)).astype(np.float32)

    probs = predict_edge_probabilities(instance, G)

    # Greedy chain decoding over the neural probability field
    unvisited = set(range(n_cust))
    curr = int(np.argmax(np.sum(probs, axis=1)))
    tour = [curr]
    unvisited.remove(curr)

    while unvisited:
        row_probs = probs[curr].copy()
        for v in tour:
            row_probs[v] = -1.0
        best_next = int(np.argmax(row_probs))
        if best_next in unvisited:
            tour.append(best_next)
            unvisited.remove(best_next)
            curr = best_next
        else:
            curr = unvisited.pop()
            tour.append(curr)

    base_key = np.zeros(n_cust, dtype=np.float32)
    for rank, cust_idx in enumerate(tour):
        base_key[cust_idx] = rank / max(1, n_cust - 1)

    n_warm = max(1, int(n_particles * warm_start_fraction))
    noise = np.random.normal(0, 0.05, size=(n_warm, n_cust))
    warm_keys = np.clip(base_key[None, :] + noise, 0.0, 1.0).astype(np.float32)
    return warm_keys