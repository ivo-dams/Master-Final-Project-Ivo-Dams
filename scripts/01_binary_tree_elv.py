# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:23:54 2026

@author: 20174931
Finalized Binary Tree Simulation Script (k=2)
Provides Figures 3.7, 3.8 and Lonely Vertices (not included).

This single script handles:
1. Exact theoretical bounds calculation for both middle vertices and the global tree.
2. Fast localized middle vertex simulation (Active Set).
3. A Full Tree Simulation that simultaneously records global expectation 
   trajectories (t <= max_t) and tracks lonely vertices at convergence.
"""

import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sympy as sp

# =============================================================================
# PART 1: EXACT CONSTANTS & THEORETICAL BOUNDS
# =============================================================================

def compute_k_ary_exact_values(k=2):
    """
    Calculates exact Symbolic Constants using SymPy by evaluating the 
    survival mechanisms of MAX-LPA at t=2.
    """
    x, y, eps = sp.symbols('x y eps')

    # Probability of a bond initiated by v
    poly_bond1 = sp.integrate(x**(k+1) * (1-(1-x**k)**(k+1)), (x, 0, eps))
    
    # Probability of a bond initiated by a neighbor
    B_sum = sum(sp.binomial(k,m) * ((x**(k+1))**m) * (y - x**(k+1))**(k-m) for m in range(0, 3))
    poly_bond2 = (k+1) * sp.integrate(sp.integrate(y**k * B_sum, (y, x, eps)), (x, 0, eps))
    poly_bond = poly_bond1 + poly_bond2
    
    # Standard Flip-flop (j=2)
    intruder_logic = sp.integrate((k-1) * (y-x**(k+1))**(k-2) * (1-y**k), (y, x, 1))
    poly_ff1 = sp.binomial(k + 1, 2) * sp.integrate(x**(2*(k+1)) * intruder_logic, (x, 0, eps))
    
    # Super Flip-flop (j>=3)
    ff_m_probs = {}
    poly_ff2 = sp.S.Zero
    for m in range(3, k+1):
        term = sp.binomial(k+1, m) * sp.integrate(x**(m*(k+1)) * ((1-x**(k+1))**(k+1-m) - (x-x**(k+1))**(k+1-m)), (x, 0, eps))
        poly_ff2 += term
        ff_m_probs[m] = float(term.subs(eps, 1))

    # Residual state (No Duplicates)
    prob_nodups = 1 - poly_bond.subs(eps, 1) - poly_ff1.subs(eps, 1) - poly_ff2.subs(eps, 1)
    
    # Compute expected label value from bond derivations
    pdf_bond = sp.diff(poly_bond, eps)
    E_bond_partial = float(sp.integrate(eps * pdf_bond, (eps, 0, 1)))
    
    # Total combined expectation at t=2
    poly_total = poly_bond + poly_ff1 + poly_ff2 + (prob_nodups * eps**(k**2 + 2*k + 2))
    pdf_total = sp.diff(poly_total, eps)
    E_mid_t2 = float(sp.integrate(eps * pdf_total, (eps, 0, 1)))

    return {
        'E_bond_partial': E_bond_partial,
        'P_FF2': float(poly_ff1.subs(eps, 1)),
        'P_FF_m': ff_m_probs,
        'P_FF_all': float(poly_ff1.subs(eps, 1) + poly_ff2.subs(eps, 1)),
        'P_ND': float(prob_nodups),
        'E_mid_t2': E_mid_t2
    }

def get_n_dt(k, t, d):
    """
    Calculates the exact structural size (number of vertices) of the 
    distance-t closed neighborhood for a vertex at distance d from the leaves.
    """
    size = 0
    h_v = min(t, d)
    size += (k**(h_v + 1) - 1) // (k - 1)
    size += t
    for i in range(1, t + 1):
        h_sib = min(t - i - 1, d + i - 1)
        if h_sib >= 0:
            # Structurally patched with // (k-1) to support mathematical generalizability
            size += (k**(h_sib + 1) - 1) // (k - 1)
    return size

def theoretical_middle_bounds(k, max_t, consts):
    """Computes the theoretical upper and lower bounds specifically for a middle vertex."""
    t_values = np.arange(1, max_t + 1)
    lower_bounds, upper_bounds = [], []
    
    # Pre-calculate the absolute floor based on distance t=2 logic
    B2_size = get_n_dt(k, 2, 2)
    denom_B2 = B2_size + 1
    LB_floor = consts['E_bond_partial'] + consts['P_FF2'] * ((2*k + 3) / denom_B2)
    for m, prob in consts['P_FF_m'].items():
        LB_floor += prob * ((m*(k+1) + 1) / denom_B2)
    
    # UPDATED: No Duplicates (ND) floor reverts to exactly t=1 closed neighborhood
    LB_floor += consts['P_ND'] * ((k + 2) / (k + 3))
    
    for t in t_values:
        if t == 1: 
            val_mid = (k + 2) / (k + 3)
            LB_mid, UB_mid = val_mid, val_mid
        elif t == 2: 
            val_mid = consts['E_mid_t2']
            LB_mid, UB_mid = val_mid, val_mid
        else:
            n_mid = get_n_dt(k, t, t)
            LB_mid = LB_floor
            # Upper bound incorporates total maximum remaining potential (n_mid / n_mid + 1)
            UB_mid = consts['E_bond_partial'] + consts['P_FF_all'] * ((n_mid - 1) / (n_mid + 1)) + consts['P_ND'] * (n_mid / (n_mid + 1))
            
        lower_bounds.append(LB_mid)
        upper_bounds.append(UB_mid)
        
    return t_values, lower_bounds, upper_bounds

def theoretical_global_bounds(max_t=5):
    """
    Computes the exact weighted global Lower and Upper bounds for the entire 
    binary tree by proportionally summing the fractional layer properties.
    """
    t_values = np.arange(1, max_t + 1)
    lower_bounds, upper_bounds = [], []
    
    # Pre-calculated exact constants for k=2
    E_A = 142 / 231
    P_FF = 3 / 280
    P_ND = 37 / 140
    middle_t2 = 0.86255411
    
    # UPDATED: Exact calculation for the new 0.8329 lower bound floor
    LB_floor_exact = E_A + (P_FF * (7/11)) + (P_ND * (4/5))
    
    for t in t_values:
        LB_total, UB_total = 0, 0
        
        # Iterating through layers that are bounded by the leaf edge
        for d in range(t):
            frac_d = 1 / (2 ** (d + 1))
            if d == 0: 
                val = 2/3 if t == 1 else (4/5 if t == 2 else 17/21)
                LB_d, UB_d = val, val
            elif d == 1: 
                val = 0.8048 if t == 2 else (109/135 if t == 3 else 17/21)
                LB_d, UB_d = val, val
            else: 
                n_dt = (2 ** t) + (2 ** (d + 1)) - 2
                LB_d = LB_floor_exact 
                UB_d = E_A + P_FF * ((n_dt - 2) / (n_dt + 1)) + P_ND * (n_dt / (n_dt + 1))
                
            LB_total += frac_d * LB_d
            UB_total += frac_d * UB_d
            
        # Handling the middle fraction that hasn't hit the boundary yet
        frac_mid = 1 / (2 ** t)
        if t == 1: 
            val_mid = 4 / 5
            LB_mid, UB_mid = val_mid, val_mid
        elif t == 2: 
            val_mid = middle_t2
            LB_mid, UB_mid = val_mid, val_mid
        else:
            n_mid = 3 * (2 ** t) - 2
            LB_mid = LB_floor_exact
            UB_mid = E_A + P_FF * ((n_mid - 2) / (n_mid + 1)) + P_ND * (n_mid / (n_mid + 1))
            
        LB_total += frac_mid * LB_mid
        UB_total += frac_mid * UB_mid
        
        lower_bounds.append(LB_total)
        upper_bounds.append(UB_total)
        
    return t_values, lower_bounds, upper_bounds


# =============================================================================
# PART 2: FAST MIDDLE VERTEX SIMULATION
# =============================================================================

def build_local_neighborhood(k, t, d):
    """Generates a localized NetworkX tree representing just the t-distance neighborhood."""
    G = nx.Graph()
    center = 0
    G.add_node(center, dist=0, h=d)
    
    node_counter = 1
    queue = [(center, d, 'center', 0)]
    
    while queue:
        curr, h, n_type, dist = queue.pop(0)
        if dist < t:
            num_up = 1 if n_type in ('center', 'ancestor') else 0
            num_down = k if n_type in ('center', 'descendant') else k - 1
            if h == 0: num_down = 0 # Cannot expand beneath the leaf boundary
                
            if num_up > 0:
                G.add_node(node_counter, dist=dist+1, h=h+1)
                G.add_edge(curr, node_counter)
                queue.append((node_counter, h+1, 'ancestor', dist+1))
                node_counter += 1
                
            for _ in range(num_down):
                G.add_node(node_counter, dist=dist+1, h=h-1)
                G.add_edge(curr, node_counter)
                queue.append((node_counter, h-1, 'descendant', dist+1))
                node_counter += 1
                
    return G, center

def simulate_middle_vertex(k, max_t, runs):
    """Simulates MAX-LPA specifically on the localized middle vertex neighborhood."""
    G, center = build_local_neighborhood(k, max_t, max_t)
    nodes = list(G.nodes())
    
    # Pre-sort nodes by distance to restrict active loops
    nodes_by_dist = {dist: [] for dist in range(max_t + 1)}
    for n in nodes: nodes_by_dist[G.nodes[n]['dist']].append(n)
        
    closed_hoods = {n: [n] + list(G.neighbors(n)) for n in nodes}
    trajectories = np.zeros((runs, max_t))
    rng = np.random.default_rng()
    
    for run_idx in tqdm(range(runs), desc="Simulating Middle Vertex", leave=False):
        vals = {node: rng.random() for node in nodes}
        for step in range(1, max_t + 1):
            new_vals = {}
            # Active tracking: nodes too far away to influence the center by max_t are ignored
            active_nodes = [n for dist in range(max_t - step + 1) for n in nodes_by_dist[dist]]
                
            if step == 1:
                # t=1 optimization: Simply find the maximum in the neighborhood
                for n in active_nodes:
                    new_vals[n] = max(vals[u] for u in closed_hoods[n])
            else:
                for n in active_nodes:
                    counts = Counter([vals[u] for u in closed_hoods[n]])
                    max_freq = max(counts.values())
                    # Apply MAX-LPA logic
                    new_vals[n] = max([v for v, c in counts.items() if c == max_freq])
            
            vals = new_vals
            trajectories[run_idx, step - 1] = vals[center]
            
    # Return both the means and the 95% margin of error (MOE)
    means = np.mean(trajectories, axis=0)
    moe = 1.96 * np.std(trajectories, axis=0, ddof=1) / np.sqrt(runs)
    return means, moe


# =============================================================================
# PART 3: COMBINED FULL TREE SIMULATION (EXPECTATION & LONELY VERTICES)
# =============================================================================

def simulate_full_tree(H=10, max_t=5, runs=1000):
    """
    Executes a global MAX-LPA simulation tracking both early expected 
    label trajectories and full convergence for lonely vertex counts.
    """
    G = nx.balanced_tree(2, H)
    N = G.number_of_nodes()
    rng = np.random.default_rng()
    
    # Pre-compute structural properties to speed up main loop
    depths = nx.single_source_shortest_path_length(G, 0)
    node_layers = [H - depths[i] for i in range(N)]
    
    open_hoods = [list(G.neighbors(i)) for i in range(N)]
    closed_hoods = [open_hoods[i] + [i] for i in range(N)]
    
    lonely_layer_counts = defaultdict(int)
    total_lonely_events = 0
    trajectories = np.zeros((runs, max_t))
    
    for run_idx in tqdm(range(runs), desc=f"Full Tree Sim (H={H})"):
        vals = rng.random(N).tolist()
        
        # Start with all nodes active at t=0
        active_nodes = set(range(N))
        step = 0
        
        while active_nodes:
            new_vals = vals.copy() 
            next_active = set()
            
            for i in active_nodes:
                counts = {}
                max_freq = 0
                best_val = -1.0
                
                # Manual loop for speed (faster than Counter inside tight loops)
                for neighbor_idx in closed_hoods[i]:
                    v = vals[neighbor_idx]
                    c = counts.get(v, 0) + 1
                    counts[v] = c
                    
                    if c > max_freq:
                        max_freq = c
                        best_val = v
                    elif c == max_freq and v > best_val:
                        best_val = v
                        
                new_vals[i] = best_val
                
                # If value changes, the node and its neighbors become active again
                if best_val != vals[i]:
                    next_active.update(closed_hoods[i])
                    
            vals = new_vals
            active_nodes = next_active
            
            # Record global tree expectation at early steps
            if step < max_t:
                trajectories[run_idx, step] = sum(vals) / N
            step += 1
            
        # Pad trajectory if tree converged exceptionally fast (before max_t)
        while step < max_t:
            trajectories[run_idx, step] = sum(vals) / N
            step += 1
    
        # --- LONELY CHECK AT CONVERGENCE ---
        found_lonely_this_run = False
        for i in range(N):
            my_val = vals[i]
            is_lonely = True
            for neighbor_idx in open_hoods[i]:
                # If ANY neighbor has a label >= the vertex label, it is NOT lonely
                if my_val <= vals[neighbor_idx]:
                    is_lonely = False
                    break
            
            if is_lonely:
                found_lonely_this_run = True
                lonely_layer_counts[node_layers[i]] += 1
        
        if found_lonely_this_run:
            total_lonely_events += 1
            
    means = np.mean(trajectories, axis=0)
    moe = 1.96 * np.std(trajectories, axis=0, ddof=1) / np.sqrt(runs)
    
    return means, moe, lonely_layer_counts, total_lonely_events, N


# =============================================================================
# PART 4: PLOTTING INTERFACES
# =============================================================================

def plot_figure_3_7(k=2, max_t=5, runs=100000):
    """Generates the Middle Vertex Graph (Figure 3.7)"""
    consts = compute_k_ary_exact_values(k)
    t_values, lower_bounds, upper_bounds = theoretical_middle_bounds(k, max_t, consts)
    sim_means, sim_moe = simulate_middle_vertex(k=k, max_t=max_t, runs=runs) 
    
    print(f"\n--- Middle Vertex Results (k={k}) ---")
    for step in range(max_t):
        t_val = step + 1
        mean_val = sim_means[step]
        moe_val = sim_moe[step]
        print(f"Middle Expected value (t={t_val}): {mean_val:.6f}")
        print(f"95% Confidence Interval: [{mean_val - moe_val:.6f}, {mean_val + moe_val:.6f}] (+/- {moe_val:.6f})\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values[1:], lower_bounds[1:], 'b--', marker='o', label="Lower Bound", linewidth=2)
    plt.plot(t_values[1:], upper_bounds[1:], 'r--', marker='o', label="Upper Bound", linewidth=2)
    plt.fill_between(t_values[1:], lower_bounds[1:], upper_bounds[1:], color='gray', alpha=0.2, label="Bounded Region")
    
    plt.errorbar(t_values[1:], sim_means[1:], marker='s', linestyle='-', 
                 color='black', markersize=8, capsize=5, capthick=2, linewidth=2, 
                 label="Simulated Expectation")
    
    plt.title('Expected Label Value of a Middle Vertex for the Binary Tree', fontsize=14)
    plt.xlabel("Time step ($t$)", fontsize=12)
    plt.ylabel("Expected Label Value", fontsize=12)
    plt.xticks(t_values[1:])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_figure_3_8_and_lonely(H=10, max_t=5, runs=10000):
    """Executes full tree sim to generate Figure 3.8 and the lonely vertex distribution."""
    start_time = time.time()
    print(f"\nStarting Full Tree Simulation (H={H}, Nodes={2**(H+1)-1}, Runs={runs})...")
    
    sim_means, sim_moe, lonely_counts, total_lonely, N = simulate_full_tree(H=H, max_t=max_t, runs=runs)
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
    
    print("\n--- Global Tree Results ---")
    for step in range(max_t):
        t_val = step + 1
        mean_val = sim_means[step]
        moe_val = sim_moe[step]
        print(f"Global Expected value (t={t_val}): {mean_val:.6f}")
        print(f"95% Confidence Interval: [{mean_val - moe_val:.6f}, {mean_val + moe_val:.6f}] (+/- {moe_val:.6f})\n")
    
    # 1. Plot Figure 3.8 (Global Bounds vs Simulation)
    t_values, lower_bounds, upper_bounds = theoretical_global_bounds(max_t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values[1:], lower_bounds[1:], 'b--', marker='o', label="Lower Bound", linewidth=2)
    plt.plot(t_values[1:], upper_bounds[1:], 'r--', marker='o', label="Upper Bound", linewidth=2)
    plt.fill_between(t_values[1:], lower_bounds[1:], upper_bounds[1:], color='gray', alpha=0.2, label="Bounded Region")
    
    plt.errorbar(t_values[1:], sim_means[1:], marker='s', linestyle='-', 
                 color='black', markersize=8, capsize=5, capthick=2, linewidth=2, 
                 label="Simulated Expectation")
    
    plt.title("Global Expected Label Value for the Binary Tree", fontsize=14)
    plt.xlabel("Time step ($t$)", fontsize=12)
    plt.ylabel("Global Expected Label Value", fontsize=12)
    plt.xticks(t_values[1:])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # 2. Plot Lonely Vertices Distribution
    print(f"\nLonely vertices occurred in {total_lonely} / {runs} runs.")
    if total_lonely > 0:
        layers = sorted(lonely_counts.keys())
        probs = [lonely_counts[l] / (N * runs) for l in layers]
        
        plt.figure(figsize=(8, 6))
        plt.bar(layers, probs, width=0.5, color='mediumorchid', edgecolor='black', alpha=0.8)
        
        plt.title(f"Lonely Vertex Layer Distribution ($k=2$, $h={H}$)", fontsize=13)
        plt.xlabel("Tree Layer ($h$)", fontsize=11)
        plt.ylabel("Probability of any Vertex being a Lonely Vertex", fontsize=11)
        plt.xticks(range(H + 1))
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("--- 1. Generating Data for Figure 3.7 (Middle Vertex) ---")
    plot_figure_3_7(k=2, max_t=5, runs=1000000)
    
    print("\n--- 2. Generating Data for Figure 3.8 and Lonely Vertices ---")
    plot_figure_3_8_and_lonely(H=15, max_t=5, runs=1000)