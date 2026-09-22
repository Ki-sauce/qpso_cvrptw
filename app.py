import time
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.baselines import gap_percent
from core.engine import solve_exact, solve_qpso, solve_random, solve_ga_baseline
from core.graph_model import load_osm_network
from core.stop_selection import choose_spread_stops
from core.traffic import apply_traffic_scenario
from core.vrp import Customer, VRPInstance, capacity_feasible
from core.heuristics import solve_dynamic_heuristic
from core.visualization import plot_map_view, plot_graph_view, plot_gantt_chart

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SIH VRP Engine", page_icon="🚚", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%; }
        .hero { padding-bottom: 1rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem; }
        .hero h1 { font-size: 2.2rem; margin-bottom: 0.2rem; letter-spacing: -0.02em; color: #111827; }
        .metric-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 1.2rem; background: #ffffff; min-height: 95px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
        .metric-label { color: #6b7280; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
        .metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; color: #111827; }
        .metric-highlight { color: #2A9D8F; } /* Green for savings */
        .section-title { font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 1rem 0; color: #111827; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="hero"><h1>Enterprise Route Optimization Dashboard</h1><p>Quantum-Inspired framework minimizing operational cost and CO2 emissions in constrained urban environments.</p></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_network(place_name):
    return load_osm_network(place_name)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Environment")
    place = st.text_input("OSM Location", "Manhattan, New York, USA")
    customers_n = st.slider("Delivery Stops", 4, 30, 10)
    vehicles = st.slider("Fleet Size", 1, 10, 4)
    capacity = st.number_input("Vehicle Capacity", value=25.0, step=1.0)
    
    st.header("2. Traffic Engine")
    traffic_mode = st.radio("Data Source", ["Simulated (Seeded)", "Live Traffic API (Mock)"])
    traffic_seed = st.number_input("Stochastic Seed", 0, 9999, 42) if "Simulated" in traffic_mode else 42
    
    st.header("3. Optimization Engine")
    objective_distance = st.slider("Distance Penalty Weight (β)", 0.0, 1.0, 0.2)
    particles = st.slider("Swarm/Population Size", 10, 100, 40)
    iterations = st.slider("Max Iterations", 20, 500, 100)

    run_btn = st.button("Initialize Dispatch Sequence", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
if 'results' not in st.session_state and not run_btn:
    st.info("👈 **Configure fleet parameters and click 'Initialize Dispatch Sequence' to begin.**")
    st.stop()

if run_btn:
    results = {}
    try:
        with st.spinner("Ingesting GIS Topology & Traffic Data..."):
            G0 = get_network(place)
            mode = 'simulated' if "Simulated" in traffic_mode else 'live'
            G = apply_traffic_scenario(G0, seed=int(traffic_seed), mode=mode)
            
            # CRITICAL FIX: Repair OSMnx edge attributes for the QPSO distance matrix
            for u, v, k, data in G.edges(keys=True, data=True):
                if 'distance_m' not in data: data['distance_m'] = float(data.get('length', 10.0))
                if 'travel_time_s' not in data: data['travel_time_s'] = data['distance_m'] / 8.33
            
            depot, customer_nodes = choose_spread_stops(
                G, customers_n, seed=int(traffic_seed)
            )
            demands = np.random.default_rng(int(traffic_seed)).integers(1, 8, size=len(customer_nodes))
            if not capacity_feasible(demands, capacity, vehicles):
                raise ValueError(
                    "The generated customer demands cannot be packed into the selected fleet. "
                    "Increase capacity or fleet size."
                )
            customers_obj = [Customer(node=n, demand=float(d)) for n, d in zip(customer_nodes, demands)]
            instance = VRPInstance(depot=depot, customers=customers_obj, vehicle_capacity=float(capacity), num_vehicles=int(vehicles))

        with st.spinner("Solving: A* Greedy Baseline..."):
            results['A*'] = solve_dynamic_heuristic(G, instance, method='astar', distance_weight=objective_distance)
        with st.spinner("Solving: QPSO (Quantum Swarm)..."):
            results['QPSO'] = solve_qpso(G, instance, particles=particles, iterations=iterations, seed=int(traffic_seed), distance_weight=objective_distance)
        with st.spinner("Solving: Genetic Algorithm..."):
            results['GA'] = solve_ga_baseline(G, instance, particles=particles, iterations=iterations, seed=int(traffic_seed), distance_weight=objective_distance)

    except Exception as exc:
        st.error(f"🚨 Execution Error: {exc}")
        st.stop()

    st.session_state.update({'results': results, 'G': G, 'instance': instance})

# -----------------------------------------------------------------------------
# Business Impact Dashboard
# -----------------------------------------------------------------------------
results = st.session_state['results']
G = st.session_state['G']
instance = st.session_state['instance']

# Calculate Business Metrics (Comparing QPSO to A* Baseline)
qpso_dist = results['QPSO'].get('distance_m', 0) / 1000
astar_dist = results['A*'].get('distance_m', 0) / 1000
dist_saved_km = max(0, astar_dist - qpso_dist)

# India Logistics Averages: 8 km/L fuel efficiency, ₹100/L diesel, 2.68 kg CO2/L
liters_saved = dist_saved_km / 8.0
rupees_saved = liters_saved * 100.0
co2_saved = liters_saved * 2.68

st.markdown('<div class="section-title" style="margin-top:0;">Fleet Impact (QPSO vs. Traditional Dispatch)</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Route Optimization</div><div class="metric-value">{qpso_dist:.1f} km <span style="font-size:0.9rem; color:#6b7280;">(from {astar_dist:.1f})</span></div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Operational Savings</div><div class="metric-value metric-highlight">₹ {rupees_saved:.0f}</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Fuel Reduction</div><div class="metric-value metric-highlight">{liters_saved:.1f} Liters</div></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="metric-card"><div class="metric-label">Carbon Offset (CO2)</div><div class="metric-value metric-highlight">↓ {co2_saved:.1f} kg</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Operational Visualization
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Operational Telemetry</div>', unsafe_allow_html=True)

c_toggle_1, c_toggle_2 = st.columns([1, 2])
with c_toggle_1: selected_algo = st.radio("Inspect Algorithm:", ["QPSO", "GA", "A*"], horizontal=True)
with c_toggle_2: view_mode = st.radio("Telemetry Mode:", ["🗺️ Spatial (GIS View)", "⏺️ Topological (Graph)", "⏱️ Temporal (Schedule Gantt)"], horizontal=True)

primary_res = results[selected_algo]

if "Spatial" in view_mode:
    fig = plot_map_view(G, primary_res, instance)
elif "Topological" in view_mode:
    fig = plot_graph_view(G, primary_res, instance)
else:
    fig = plot_gantt_chart(G, primary_res, instance)
    
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -----------------------------------------------------------------------------
# Academic / Technical Proof
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Engine Convergence & Benchmarking</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 1.2])

with col_a:
    comp_data = []
    for key, res in results.items():
        dist_km = res.get('distance_m', 0) / 1000.0
        time_min = res.get('travel_time_s', 0) / 60.0
        comp_data.append({
            "Methodology": res['algorithm'],
            "Total Cost": res['score'],
            "Distance (km)": dist_km,
            "Travel Time (min)": time_min
        })
    st.dataframe(
        pd.DataFrame(comp_data).sort_values("Total Cost").style.format({
            "Total Cost": "{:.2f}", 
            "Distance (km)": "{:.2f} km", 
            "Travel Time (min)": "{:.1f} min"
        }), 
        use_container_width=True, 
        hide_index=True
    )
with col_b:
    fig_conv, ax = plt.subplots(figsize=(8, 3.8))
    colors = {'QPSO': '#D90429', 'Genetic Algorithm': '#F4A261', 'A* Constructive': '#111827'}
    
    for key, res in results.items():
        color = colors.get(res['algorithm'], '#888888')
        if 'history' in res:
            if res['algorithm'] == 'A* Constructive':
                ax.axhline(y=res["score"], color=color, linestyle='--', linewidth=2.0, label=res['algorithm'])
            else:
                ax.plot(res["history"], color=color, linewidth=2.5, label=res['algorithm'])
            
    ax.set_xlabel("Iteration Epoch")
    ax.set_ylabel("Global Objective Cost")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False, fontsize=9)
    fig_conv.tight_layout()
    st.pyplot(fig_conv)
