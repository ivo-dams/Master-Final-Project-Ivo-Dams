# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:37:25 2026

@author: 20174931

Scale-Free Convergence Time Paradox Script

Simulates MAX-LPA convergence times on the Giant Connected Component (LCC)
of Scale-Free networks across varying graph sizes (N) and power-law exponents (tau).
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: GRAPH GENERATION (LCC)
# =============================================================================

def generate_scale_free_giant_component(n, tau=2.5):
    """Generates the LCC of a scale-free graph via the erased configuration model."""
    rng = np.random.default_rng()
    while True:
        seq = rng.zipf(tau, n)
        seq = np.clip(seq, 1, n-1)
        if sum(seq) % 2 == 0:
            break
            
    G_multi = nx.configuration_model(seq)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc).copy()
    
    return nx.convert_node_labels_to_integers(G_giant)

# =============================================================================
# PART 2: ULTRA-OPTIMIZED MAX-LPA SIMULATION
# =============================================================================

def run_propagation(G, max_steps=2000):
    """
    Active-set synchronous label propagation.
    Uses native counts and list comprehension for maximum pure-Python execution speed.
    """
    n_nodes = G.number_of_nodes()
    
    # Pre-compute closed neighborhoods as lists of indices for O(1) lookups
    closed_hoods = [[i] + list(G.neighbors(i)) for i in range(n_nodes)]
    vals = [np.random.random() for _ in range(n_nodes)]
    
    seen_states = set()
    steps = 0
    active_nodes = set(range(n_nodes))
    
    while steps < max_steps:
        # Delayed cycle detection: Skip early states to save memory/processing overhead
        if steps > 3:
            current_state = tuple(vals)
            if current_state in seen_states:
                return steps, True  # Infinite cycle detected
            seen_states.add(current_state)
            
        steps += 1
        pending_updates = []
        changed_nodes = set()
        
        for n in active_nodes:
            hood_vals = [vals[u] for u in closed_hoods[n]]
            
            # Fast majority vote: max() picks highest count, breaks ties with highest label value
            best_val = max(set(hood_vals), key=lambda v: (hood_vals.count(v), v))
            
            if best_val != vals[n]:
                pending_updates.append((n, best_val))
                changed_nodes.add(n)
        
        if not pending_updates:
            return steps, False  # Successfully converged
        
        # Commit updates synchronously
        for n, new_val in pending_updates:
            vals[n] = new_val
            
        # Rebuild active set: only nodes near changes need re-evaluation
        active_nodes = set()
        for n in changed_nodes:
            active_nodes.update(closed_hoods[n])

    return steps, True # Hit max_steps bounds

# =============================================================================
# PART 3: MASTER SIMULATION
# =============================================================================

def simulate_scale_free_convergence(tau_values=[2.1, 2.5, 2.9], 
                                    n_targets=[1000, 2000, 4000, 8000, 16000, 32000, 64000], 
                                    runs=50):
    """
    Simulates MAX-LPA to convergence across varying scale-free exponents and graph sizes.
    """
    global_start_time = time.time()
    
    # Store multiple x-axis representations to give plotting flexibility
    results = {tau: {"n_target": [], "n_actual": [], "ln_n": [], "log2_n": [], "t_mean": [], "moe": []} for tau in tau_values}
    total_cycles = 0
    
    for tau in tau_values:
        print(f"\nSimulating \u03C4={tau}...")
        
        for target_n in n_targets:
            steps_list = []
            actual_n_list = []
            
            for _ in tqdm(range(runs), desc=f"Target N={target_n}", leave=False):
                G = generate_scale_free_giant_component(target_n, tau)
                
                # Record the true size of the LCC
                actual_n = G.number_of_nodes()
                actual_n_list.append(actual_n)
                
                s, is_cycle = run_propagation(G)
                if is_cycle:
                    total_cycles += 1
                steps_list.append(s)
            
            # Statistical aggregation
            mean_actual_n = np.mean(actual_n_list)
            mean_steps = np.mean(steps_list)
            std = np.std(steps_list, ddof=1) if runs > 1 else 0
            sem = std / np.sqrt(runs) if runs > 0 else 0
            moe = 1.96 * sem
            
            results[tau]["n_target"].append(target_n)
            results[tau]["n_actual"].append(mean_actual_n)
            results[tau]["ln_n"].append(np.log(mean_actual_n))
            results[tau]["log2_n"].append(np.log2(mean_actual_n))
            results[tau]["t_mean"].append(mean_steps)
            results[tau]["moe"].append(moe)
            
    total_time = time.time() - global_start_time
    print(f"\nSimulation complete in {total_time:.2f} seconds.")
    print(f"Total infinite flip-flop cycles detected: {total_cycles}")
    
    return results

# =============================================================================
# PART 4: PLOTTING INTERFACE
# =============================================================================

def plot_convergence_time(results, x_axis='log2'):
    """
    Plots the convergence paradox.
    Parameters:
      x_axis (str): 'ln' or 'log2'
    """
    print(f"\n{'='*60}")
    print(f" 95% CONFIDENCE INTERVALS (X-Axis: {x_axis.upper()})")
    print(f"{'='*60}")
    
    # Console Output
    for tau, data in results.items():
        print(f"\u03C4 = {tau}:")
        for i in range(len(data["n_target"])):
            target_n = data["n_target"][i]
            actual_n = data["n_actual"][i]
            mean_val = data["t_mean"][i]
            moe_val = data["moe"][i]
            
            if x_axis == 'ln':
                print(f"  Target N={target_n:<7d} | LCC Avg N={int(actual_n):<7d} | ln(N)={data['ln_n'][i]:.3f} | Mean Time: {mean_val:.3f} ± {moe_val:.3f}")
            else: # Default to log2
                print(f"  Target N={target_n:<7d} | LCC Avg N={int(actual_n):<7d} | log2(N)={data['log2_n'][i]:.3f} | Mean Time: {mean_val:.3f} ± {moe_val:.3f}")
        print("-" * 40)
    print("="*60 + "\n")

    # Styling consistent with thesis aesthetics
    colors = {2.1: 'steelblue', 2.5: 'firebrick', 2.9: 'forestgreen'}
    markers = {2.1: 'v', 2.5: 's', 2.9: 'o'}
    
    plt.figure(figsize=(10, 6))
    
    # Select the requested data vector for the X-axis
    for tau, data in results.items():
        means = np.array(data["t_mean"])
        
        if x_axis == 'ln':
            x_data = data["ln_n"]
        else: # Default log2
            x_data = data["log2_n"]
            
        plt.plot(x_data, means, marker=markers.get(tau, 'o'), linestyle='-', 
                 color=colors.get(tau, 'black'), markersize=8, linewidth=2, 
                 label=f"$\\tau={tau}$")
        
    # Dynamic axis labeling based on selection
    if x_axis == 'ln':
        plt.xlabel(r"Graph Size $\ln(n)$", fontsize=12)
    else: # log2
        plt.xlabel(r"Graph Size $\log_2(n)$", fontsize=12)

    plt.title("Convergence Time vs Scale-Free Graph Size (LCC)", fontsize=14)
    plt.ylabel(r"Mean Number of Time Steps to Convergence", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Feel free to adjust the targets and runs depending on your available execution time.
    res = simulate_scale_free_convergence(
        tau_values=[2.1, 2.5, 2.9], 
        #n_targets=[1000, 2000, 4000, 8000, 16000, 32000, 64000],
        n_targets=[1000, 2000, 4000, 8000, 16000],
        runs=1000
    )
    
    print("\n--- Generating Figure (log2 scale) ---")
    plot_convergence_time(res, x_axis='log2')
    
    print("\n--- Generating Figure (ln scale) ---")
    plot_convergence_time(res, x_axis='ln')