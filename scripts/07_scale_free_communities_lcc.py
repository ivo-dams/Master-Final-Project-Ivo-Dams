# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:27:57 2026

@author: 20174931

Script 2: Structural Analysis (LCC Only)

Generates the Giant Connected Component of a Scale-Free Graph.
Calculates and plots the 2x2 Step Grids, Community PMF/CCDF, 
Degree/Label Correlations, and Lonely Vertices distributions.
"""

import numpy as np
import networkx as nx
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# PART 1: GRAPH GENERATION & SMOOTHING TOOLS
# =============================================================================

def generate_scale_free_giant_component(n, tau=2.5):
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

def log_kernel_smoother(x, y, counts, bandwidth=0.2):
    x_log = np.log10(x)
    y_smooth = np.zeros_like(y, dtype=float)
    for i, x0 in enumerate(x_log):
        distance_weights = np.exp(-0.5 * ((x_log - x0) / bandwidth)**2)
        total_weights = distance_weights * counts
        if np.sum(total_weights) > 0:
            y_smooth[i] = np.sum(total_weights * y) / np.sum(total_weights)
        else:
            y_smooth[i] = 0
    return y_smooth

# =============================================================================
# PART 2: OPTIMIZED MAX-LPA (STRUCTURAL ONLY)
# =============================================================================

def run_unified_max_lpa_fast_structural(G, rng, max_t=5):
    N = G.number_of_nodes()
    
    closed_hoods = [[i] + list(G.neighbors(i)) for i in range(N)]
    initial_degrees = [G.degree(i) for i in range(N)]
    
    initial_labels = [rng.random() for _ in range(N)]
    vals = initial_labels.copy()
    
    step_deg_sums = {t: defaultdict(float) for t in range(1, max_t + 1)}
    step_deg_counts = {t: defaultdict(int) for t in range(1, max_t + 1)}
    
    history_minus_1 = None
    history_minus_2 = vals.copy()
    
    step = 0
    final_vals = None
    is_cycle = False
    
    active_nodes = set(range(N))
    
    while True:
        step += 1
        new_vals = vals.copy()
        next_active = set()
        changed = False
        
        for n in active_nodes:
            counts = {}
            max_freq = 0
            best_val = -1.0
            
            for u in closed_hoods[n]:
                v = vals[u]
                c = counts.get(v, 0) + 1
                counts[v] = c
                
                if c > max_freq:
                    max_freq = c
                    best_val = v
                elif c == max_freq and v > best_val:
                    best_val = v
            
            if best_val != vals[n]:
                new_vals[n] = best_val
                changed = True
                next_active.update(closed_hoods[n])
                
        history_minus_2 = history_minus_1
        history_minus_1 = vals
        vals = new_vals
        active_nodes = next_active
        
        if step <= max_t:
            for idx_node, val in enumerate(vals):
                d = initial_degrees[idx_node]
                step_deg_sums[step][d] += val
                step_deg_counts[step][d] += 1
                
        if not changed:
            final_vals = new_vals
            break
            
        if history_minus_2 is not None and new_vals == history_minus_2:
            final_vals = history_minus_1
            is_cycle = True
            break
            
        if step > 150:
            final_vals = vals
            is_cycle = True
            break

    # Pad expected value correlations if convergence hit before max_t
    pad_step = len(step_deg_sums) 
    while pad_step < max_t:
        pad_step += 1
        for idx_node, val in enumerate(final_vals):
            d = initial_degrees[idx_node]
            step_deg_sums[pad_step][d] += val
            step_deg_counts[pad_step][d] += 1

    final_counts = Counter(final_vals)
    community_sizes = list(final_counts.values())
    
    val_to_node = {initial_labels[i]: i for i in range(N)}
    lonely_degrees = [initial_degrees[i] for i, lbl in enumerate(final_vals) if final_counts[lbl] == 1]

    fates_all = [{'degree': initial_degrees[i], 'init_label': initial_labels[i], 'size': final_counts.get(initial_labels[i], 0)} for i in range(N)]
    fates_survivors = [{'degree': initial_degrees[val_to_node[lbl]], 'init_label': lbl, 'size': size} for lbl, size in final_counts.items()]

    return community_sizes, fates_all, fates_survivors, lonely_degrees, is_cycle, step_deg_sums, step_deg_counts

# =============================================================================
# PART 3: MASTER SIMULATION AGGREGATOR
# =============================================================================

def simulate_master_structural(num_nodes=3000, tau_values=[2.1, 2.5, 2.9], runs=50, max_t=4):
    start_time = time.time()
    rng = np.random.default_rng()
    
    cycle_counts = {tau: 0 for tau in tau_values}
    pmf_frequencies = {tau: defaultdict(int) for tau in tau_values}
    all_deg_sizes = {tau: defaultdict(list) for tau in tau_values}
    all_label_sizes = {tau: defaultdict(list) for tau in tau_values}
    surv_deg_sizes = {tau: defaultdict(list) for tau in tau_values}
    surv_label_sizes = {tau: defaultdict(list) for tau in tau_values}
    all_lonely_degrees = {tau: [] for tau in tau_values}
    
    agg_step_deg_sums = {tau: {t: defaultdict(float) for t in range(1, max_t + 1)} for tau in tau_values}
    agg_step_deg_counts = {tau: {t: defaultdict(int) for t in range(1, max_t + 1)} for tau in tau_values}
    
    label_bins = np.linspace(0.0, 1.0, 101)

    print(f"Starting structural simulation on LCC (N={num_nodes}, Runs={runs})...")
    
    for tau in tau_values:
        print(f"\nSimulating tau = {tau}...")
        for run_idx in tqdm(range(runs), desc=f"Networks", leave=False):
            G = generate_scale_free_giant_component(num_nodes, tau)
            
            (sizes, f_all, f_surv, lonely_degs, is_cycle, 
             step_deg_sums, step_deg_counts) = run_unified_max_lpa_fast_structural(G, rng, max_t)
            
            if is_cycle: cycle_counts[tau] += 1
            all_lonely_degrees[tau].extend(lonely_degs)
            
            for size in sizes: pmf_frequencies[tau][size] += 1
                
            for fate in f_all:
                all_deg_sizes[tau][fate['degree']].append(fate['size'])
                b_idx = np.clip(np.digitize(fate['init_label'], label_bins) - 1, 0, 99)
                all_label_sizes[tau][b_idx].append(fate['size'])
                
            for fate in f_surv:
                surv_deg_sizes[tau][fate['degree']].append(fate['size'])
                b_idx = np.clip(np.digitize(fate['init_label'], label_bins) - 1, 0, 99)
                surv_label_sizes[tau][b_idx].append(fate['size'])

            for t_step in range(1, max_t + 1):
                for d, s_val in step_deg_sums[t_step].items():
                    agg_step_deg_sums[tau][t_step][d] += s_val
                    agg_step_deg_counts[tau][t_step][d] += step_deg_counts[t_step][d]

    print(f"\nSimulation complete in {time.time() - start_time:.2f} seconds.")
    
    return (pmf_frequencies, all_deg_sizes, all_label_sizes, surv_deg_sizes, 
            surv_label_sizes, all_lonely_degrees, cycle_counts, tau_values, 
            num_nodes, runs, max_t, agg_step_deg_sums, agg_step_deg_counts)

# =============================================================================
# PART 4: PLOTTING
# =============================================================================

def plot_structural_figures(data_tuple):
    (pmf_freqs_dict, all_deg_dict, all_lbl_dict, surv_deg_dict, surv_lbl_dict, 
     lonely_degs_dict, cycle_counts, tau_values, n, runs, max_t, 
     agg_step_deg_sums, agg_step_deg_counts) = data_tuple

    print("\n" + "="*50)
    print(" 2-CYCLE CONVERGENCE PROBABILITIES")
    print("="*50)
    for tau, count in cycle_counts.items():
        print(f"  tau = {tau:<4.1f} | Probability: {count / runs:.4f} ({count}/{runs} runs)")
    print("="*50)

    for tau in tau_values:
        # --- 1. 2x2 Grid: Expected Label Value vs Node Degree Over Time ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Expected Label Value vs. Vertex Degree ($\u03C4$={tau})", fontsize=16, y=0.98)
        axes = axes.flatten()
        
        for step in range(1, min(max_t, 4) + 1):
            ax = axes[step - 1]
            degrees = np.array(sorted(list(agg_step_deg_sums[tau][step].keys())))
            avg_vals = np.array([agg_step_deg_sums[tau][step][d] / agg_step_deg_counts[tau][step][d] for d in degrees])
            counts_array = np.array([agg_step_deg_counts[tau][step][d] for d in degrees])
            
            ax.scatter(degrees, avg_vals, alpha=0.4, s=25, color='royalblue', edgecolors='none', label='Simulated Data')
            
            if step >= 2:
                y_smooth = log_kernel_smoother(degrees[1:], avg_vals[1:], counts_array[1:], bandwidth=0.1)
                slice_idx = int(len(y_smooth) * 0.8)
                if slice_idx > 0:
                    ax.plot(degrees[1:1+slice_idx], y_smooth[:slice_idx], color='firebrick', linewidth=2.5, linestyle='-', label='Weighted Trend')
                
            if step == 1:
                x_exact = np.logspace(np.log10(min(degrees)), np.log10(max(degrees)), 200)
                y_exact = (x_exact + 1) / (x_exact + 2)
                ax.plot(x_exact, y_exact, color='black', linewidth=2, linestyle='--', label=r"Exact Analytical: $\frac{d+1}{d+2}$")
            
            ax.set_title(f"Time Step t={step}", fontsize=13)
            ax.set_xlabel("Degree $d$ (Log Scale)", fontsize=11)
            ax.set_ylabel(r"Expected Label Value", fontsize=11)
            ax.set_xscale('log')
            
            if step == 2: ax.set_ylim([-0.05, 1.05])
            else: ax.set_ylim([0.65, 1.025])
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right', fontsize=10)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 2. Structural Community Plots ---
        pmf_freqs = pmf_freqs_dict[tau]
        sizes = np.array(sorted(list(pmf_freqs.keys())))
        
        if len(sizes) > 0:
            freqs = np.array([pmf_freqs[s] for s in sizes])
            pmf = freqs / np.sum(freqs)
            
            # PMF Log-Log
            plt.figure(figsize=(9, 6))
            plt.scatter(sizes, pmf, alpha=0.7, s=40, color='darkslateblue', edgecolors='black', linewidths=0.5)
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"Scale-Free Community Size Distribution PMF ($\u03C4$={tau})", fontsize=13)
            plt.xlabel("Community Size ($S$)", fontsize=11)
            plt.ylabel("Proportion", fontsize=11)
            plt.grid(True, which="both", linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.show()
            
            # CCDF Log-Log
            ccdf = 1.0 - np.cumsum(pmf)
            plt.figure(figsize=(9, 6))
            plt.step(sizes[:-1], ccdf[:-1], color='teal', linewidth=2, where='post', alpha=0.8)
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"Scale-Free Community Size CCDF ($\u03C4$={tau})", fontsize=13)
            plt.xlabel("Community Size ($S$)", fontsize=11)
            plt.ylabel(r"Proportion", fontsize=11)
            plt.grid(True, which="both", linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.show()

        def process_matrices(deg_dict, lbl_dict):
            u_deg = np.array(sorted(list(deg_dict.keys())))
            avg_deg = np.array([np.mean(deg_dict[d]) for d in u_deg])
            c_deg = np.array([len(deg_dict[d]) for d in u_deg])
            a_bins = np.array(sorted(list(lbl_dict.keys())))
            b_cent = np.array([0.005 + b*0.01 for b in a_bins])
            avg_bin = np.array([np.mean(lbl_dict[b]) for b in a_bins])
            return u_deg, avg_deg, c_deg, b_cent, avg_bin

        # --- 3. Degree & Label Correlations (All Nodes vs Survivors) ---
        u_deg, avg_deg, c_deg, b_cent, avg_bin = process_matrices(all_deg_dict[tau], all_lbl_dict[tau])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(u_deg, avg_deg, alpha=0.5, s=30, color='royalblue', edgecolors='none', label='Empirical Expected Size')
        if len(u_deg) > 2:
            y_smooth = log_kernel_smoother(u_deg, avg_deg, c_deg, bandwidth=0.15)
            cutoff = int(len(u_deg) * 0.8)
            plt.plot(u_deg[:cutoff], y_smooth[:cutoff], color='firebrick', linewidth=2.5, label='Weighted Trend')
        plt.title(f"Expected Community Size vs. Vertex Degree (All Labels, $\u03C4={tau}$)", fontsize=13)
        plt.xlabel("Vertex Degree $d$ (Log Scale)", fontsize=11)
        plt.ylabel("Expected Community Size (Log Scale)", fontsize=11)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(b_cent, avg_bin, alpha=0.6, s=35, color='darkorange', edgecolors='none')
        if len(avg_bin) > 5:
            y_smooth_b = np.convolve(avg_bin, np.ones(5)/5, mode='same')
            plt.plot(b_cent[2:-2], y_smooth_b[2:-2], color='darkred', linewidth=2.5)
        plt.title(f"Expected Community Size vs. Initial Label Value (All Labels, $\u03C4={tau}$)", fontsize=13)
        plt.xlabel(r"Initial Label Value $\ell_v(0)$", fontsize=11)
        plt.ylabel("Expected Final Community Size (Log Scale)", fontsize=11)
        plt.yscale('log')
        plt.xlim([-0.02, 1.02])
        plt.grid(True, which="both", linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        u_deg, avg_deg, c_deg, b_cent, avg_bin = process_matrices(surv_deg_dict[tau], surv_lbl_dict[tau])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(u_deg, avg_deg, alpha=0.5, s=30, color='royalblue', edgecolors='none')
        if len(u_deg) > 2:
            y_smooth = log_kernel_smoother(u_deg, avg_deg, c_deg, bandwidth=0.15)
            plt.plot(u_deg, y_smooth, color='firebrick', linewidth=2.5)
        plt.title(f"Expected Community Size vs. Vertex Degree (Survivors, $\u03C4={tau}$)", fontsize=13)
        plt.xlabel("Vertex Degree $d$ (Log Scale)", fontsize=11)
        plt.ylabel("Expected Final Community Size (Log Scale)", fontsize=11)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(b_cent, avg_bin, alpha=0.6, s=35, color='darkorange', edgecolors='none')
        if len(avg_bin) > 5:
            y_smooth_b = np.convolve(avg_bin, np.ones(5)/5, mode='same')
            plt.plot(b_cent[2:-2], y_smooth_b[2:-2], color='darkred', linewidth=2.5)
        plt.title(f"Expected Community Size vs. Initial Label Value (Survivors, $\u03C4={tau}$)", fontsize=13)
        plt.xlabel(r"Initial Label Value $\ell_v(0)$", fontsize=11)
        plt.ylabel("Expected Final Community Size (Log Scale)", fontsize=11)
        plt.yscale('log')
        plt.xlim([-0.02, 1.02])
        plt.grid(True, which="both", linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # --- 4. Lonely Vertices Local Distribution ---
        lonely_degs = lonely_degs_dict[tau]
        if lonely_degs:
            deg_counts = Counter(lonely_degs)
            unique_degs = np.array(sorted(list(deg_counts.keys())))
            probs = np.array([deg_counts[d] for d in unique_degs]) / (n * runs)
            
            print("\n" + "="*50)
            print(f" LONELY VERTEX DEGREE DISTRIBUTION (\u03C4={tau})")
            print("="*50)
            for d, p in zip(unique_degs, probs):
                print(f"  Degree={d:<5d} | Probability: {p:.6f}")
            print("="*50 + "\n")
            
            plt.figure(figsize=(8, 6))
            bar_widths = unique_degs * 0.15
            plt.bar(unique_degs, probs, width=bar_widths, color='mediumorchid', edgecolor='black', alpha=0.8)
            plt.xscale('log')
            plt.title(f"Lonely Vertex Degree Distribution ($\u03C4={tau}$)", fontsize=13)
            plt.xlabel("Vertex Degree $d$ (Log Scale)", fontsize=11)
            plt.ylabel("Probability of any Vertex being a Lonely Vertex", fontsize=11)
            plt.grid(True, which="both", linestyle=':', alpha=0.5)
            plt.tight_layout()
            plt.show()

    # ---------------------------------------------------------
    # 5. Global Cycle & Lonely Probabilities (Aggregated by Tau)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    taus = [str(t) for t in cycle_counts.keys()]
    cycle_probs = [count / runs for count in cycle_counts.values()]
    plt.bar(taus, cycle_probs, color='crimson', edgecolor='black', alpha=0.8, width=0.5)
    plt.title("Probability of 2-Cycle Convergence by Scale-Free Exponent", fontsize=13)
    plt.xlabel(r"Scale-Free Exponent ($\tau$)", fontsize=11)
    plt.ylabel("Probability of 2-Cycle", fontsize=11)
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sorted_taus = sorted(list(tau_values))
    aggregate_probs = [len(lonely_degs_dict[t]) / (n * runs) for t in sorted_taus]
    plt.bar([str(t) for t in sorted_taus], aggregate_probs, color='darkorchid', edgecolor='black', alpha=0.8, width=0.5)
    plt.title("Lonely Vertex Probability by Scale-Free Exponent", fontsize=13)
    plt.xlabel(r"Scale-Free Exponent ($\tau$)", fontsize=11)
    plt.ylabel("Probability of any Vertex being a Lonely Vertex", fontsize=11)
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    data = simulate_master_structural(num_nodes=100000, tau_values=[2.1, 2.5, 2.9], runs=1000, max_t=5)
    plot_structural_figures(data)