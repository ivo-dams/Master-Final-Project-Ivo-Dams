# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:22:34 2026

@author: 20174931

Combined Convergence Time Paradox Script (k-ary Trees)

Provides Data and Figures for the Convergence Paradox (Figures 6.1, 6.2).

This script implements:
1. Ultra-optimized, active-set MAX-LPA tracking steps to convergence.
2. Delayed cycle detection.
3. A unified plotting interface allowing the user to plot convergence time 
   against tree height (h), natural log (ln(n)), or base-2 log (log2(n)).
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: ULTRA-OPTIMIZED MAX-LPA SIMULATION
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
# PART 2: GRAPH SIZING & EXPERIMENT EXECUTION
# =============================================================================

def get_max_h(k, max_n):
    """Calculates the maximum complete tree height h that keeps total nodes <= max_n."""
    h = 1 
    while True:
        n = (k**(h+1) - 1) // (k - 1)
        if n > max_n:
            return max(1, h - 1)
        h += 1

def simulate_convergence_paradox(k_values=[2, 3, 4, 5, 9], max_n=100000, runs=1000):
    """
    Simulates MAX-LPA to convergence across varying branching factors and graph sizes.
    Records steps, infinite cycles, and statistical margins of error.
    """
    global_start_time = time.time()
    
    # Store multiple x-axis representations to give plotting flexibility later
    results = {k: {"h": [], "n": [], "ln_n": [], "log2_n": [], "t_mean": [], "moe": []} for k in k_values}
    total_cycles = 0
    
    for k in k_values:
        max_h = get_max_h(k, max_n)
        print(f"\nSimulating k={k} up to height h={max_h}...")
        
        for h in range(1, max_h + 1):
            G = nx.balanced_tree(k, h)
            n = G.number_of_nodes()
            
            steps_list = []
            
            for _ in tqdm(range(runs), desc=f"h={h}, n={n}", leave=False):
                s, is_cycle = run_propagation(G)
                if is_cycle:
                    total_cycles += 1
                # Even if it cycles, we record the steps it took to lock into the cycle
                steps_list.append(s)
            
            # Statistical aggregation
            mean = np.mean(steps_list)
            std = np.std(steps_list, ddof=1) if runs > 1 else 0
            sem = std / np.sqrt(runs) if runs > 0 else 0
            moe = 1.96 * sem
            
            results[k]["h"].append(h)
            results[k]["n"].append(n)
            results[k]["ln_n"].append(np.log(n))
            results[k]["log2_n"].append(np.log2(n))
            results[k]["t_mean"].append(mean)
            results[k]["moe"].append(moe)
            
    total_time = time.time() - global_start_time
    print(f"\nSimulation complete in {total_time:.2f} seconds.")
    print(f"Total infinite flip-flop cycles detected: {total_cycles}")
    
    return results


# =============================================================================
# PART 3: PLOTTING INTERFACE
# =============================================================================

def plot_convergence_time(results, x_axis='log2'):
    """
    Plots the convergence paradox.
    Parameters:
      x_axis (str): 'height (Figure 6.1)' or 'log2' (Figure 6.2)
    """
    print(f"\n{'='*60}")
    print(f" 95% CONFIDENCE INTERVALS (X-Axis: {x_axis.upper()})")
    print(f"{'='*60}")
    
    # Console Output
    for k, data in results.items():
        print(f"k = {k}:")
        for i in range(len(data["h"])):
            h_val = data["h"][i]
            n_val = data["n"][i]
            mean_val = data["t_mean"][i]
            moe_val = data["moe"][i]
            
            if x_axis == 'height':
                print(f"  Height={h_val:2d} | Mean Time: {mean_val:.3f} ± {moe_val:.3f}")
            elif x_axis == 'ln':
                print(f"  N={n_val:<7d} | ln(N)={data['ln_n'][i]:.3f} | Mean Time: {mean_val:.3f} ± {moe_val:.3f}")
            else: # Default to log2
                print(f"  N={n_val:<7d} | log2(N)={data['log2_n'][i]:.3f} | Mean Time: {mean_val:.3f} ± {moe_val:.3f}")
        print("-" * 40)
    print("="*60 + "\n")

    # Styling consistent with thesis aesthetics
    colors = {2: 'steelblue', 3: 'firebrick', 4: 'forestgreen', 5: 'darkorange', 9: 'black'}
    markers = {2: 'v', 3: '^', 4: 's', 5: 'D', 9: 'o'}
    
    plt.figure(figsize=(10, 6))
    
    # Select the requested data vector for the X-axis
    for k, data in results.items():
        means = np.array(data["t_mean"])
        
        if x_axis == 'height':
            x_data = data["h"]
        elif x_axis == 'ln':
            x_data = data["ln_n"]
        else: # Default log2
            x_data = data["log2_n"]
            
        plt.plot(x_data, means, marker=markers.get(k, 'o'), linestyle='-', 
                 color=colors.get(k, 'black'), markersize=8, linewidth=2, 
                 label=f"$k={k}$")
        
    # Dynamic axis labeling based on selection
    if x_axis == 'height':
        plt.xlabel(r"Tree Height ($h$)", fontsize=12)
        plt.title("Convergence Time vs Tree Height", fontsize=14)
        max_h_overall = max([max(d["h"]) for d in results.values()])
        plt.xticks(range(1, max_h_overall + 1))
    elif x_axis == 'ln':
        plt.xlabel(r"$\ln(n)$", fontsize=12)
        plt.title("Convergence Time vs Graph Size", fontsize=14)
    else: # log2
        plt.xlabel(r"$\log_2(n)$", fontsize=12)
        plt.title("Convergence Time vs Graph Size", fontsize=14)

    plt.ylabel(r"Mean Number of Time Steps to Convergence", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Recommended production values: max_n=100000, runs=1000
    res = simulate_convergence_paradox(k_values=[2, 3, 4, 5, 9], max_n=100000, runs=1000)
    
    print("\n--- Generating Figure 6.1 (tree height scale) ---")
    plot_convergence_time(res, x_axis='height')
    
    print("\n--- Generating Figure 6.2 (log2 scale) ---")
    plot_convergence_time(res, x_axis='log2')
