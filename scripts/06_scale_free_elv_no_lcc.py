# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:13:45 2026

@author: 20174931

Script 1: Expected Label Values (Entire Graph)

Generates the entire Scale-Free graph (no LCC extraction).
Calculates and plots global expected trajectories (IMMEDIATELY after each tau) 
and the final converged Expected Label Value vs Scale-Free Exponent sweep.
"""

import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import zeta

# =============================================================================
# PART 1: THEORETICAL BOUNDS (ENTIRE GRAPH)
# =============================================================================

class MaxLpaBounds:
    def __init__(self, tau):
        self.tau = tau
        self.zeta_tau = zeta(tau, 1)
        self.zeta_tau_minus_1 = zeta(tau - 1, 1)

    def prob_degree(self, d):
        """Standard degree distribution P(D = d) in the base network."""
        if d < 1: return 0.0
        return (d ** -self.tau) / self.zeta_tau

    def prob_forward_degree(self, k):
        """Forward degree distribution P(D* = k) in the base network."""
        if k < 0: return 0.0
        return ((k + 1) ** (1 - self.tau)) / self.zeta_tau_minus_1

    def bound_degree_1(self, cutoff=100000):
        """Rigorous lower bound for d=1 with finite-size normalization."""
        # Step 1: Calculate the total probability mass up to the cutoff
        total_prob_mass = 0.0
        for k in range(cutoff):
            total_prob_mass += self.prob_forward_degree(k)
            
        # Step 2: Calculate the expected penalty using normalized probabilities
        expected_penalty = 0.0
        for k in range(cutoff):
            # Normalize the probability so the truncated sum equals 1.0
            normalized_prob = self.prob_forward_degree(k) / total_prob_mass
            
            # Apply the penalty logic
            expected_penalty += (1.0 / (k + 3)) * normalized_prob
            
        return 1.0 - expected_penalty

    def bound_degree_2(self):
        """Degree 2 vertices almost surely do not decrease (bound is t=1 expectation)."""
        return 3.0 / 4.0

    def bound_degree_ge_3(self, d):
        """Rigorous absolute mathematical floor for d >= 3."""
        return 0.5

    def compute_t1_expectation(self, max_degree_cutoff=100000):
        """Exact theoretical expected label at t=1."""
        exp = 0.0
        cumulative_prob = 0.0
        for d in range(1, max_degree_cutoff + 1):
            prob_d = self.prob_degree(d)
            cumulative_prob += prob_d
            exp += prob_d * ((d + 1.0) / (d + 2.0))
        # Add remaining infinite tail mass (assuming extreme boundary limit of 1.0)
        exp += (1.0 - cumulative_prob) * 1.0 
        return exp

    def compute_general_bounds(self, max_degree_cutoff=100000):
        """Theoretical expected lower bound for t >= 2."""
        expectation = 0.0
        cumulative_prob = 0.0
        
        for d in range(1, max_degree_cutoff + 1):
            prob_d = self.prob_degree(d)
            cumulative_prob += prob_d
            
            if d == 1:
                b_d = self.bound_degree_1()
            elif d == 2:
                b_d = self.bound_degree_2()
            else:
                b_d = self.bound_degree_ge_3(d)
                
            expectation += prob_d * b_d
            
        # Add remaining infinite tail mass conservatively
        expectation += (1.0 - cumulative_prob) * 0.5
        return expectation

# =============================================================================
# PART 2: GRAPH GENERATION (ENTIRE GRAPH)
# =============================================================================

def generate_scale_free_graph(n, tau=2.5):
    rng = np.random.default_rng()
    while True:
        seq = rng.zipf(tau, n)
        seq = np.clip(seq, 1, n-1)
        if sum(seq) % 2 == 0:
            break
            
    G_multi = nx.configuration_model(seq)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return nx.convert_node_labels_to_integers(G)

# =============================================================================
# PART 3: OPTIMIZED UNIFIED MAX-LPA SIMULATION (EXPECTED VALUES ONLY)
# =============================================================================

def run_unified_max_lpa_fast_expected(G, rng, max_t=5):
    N = G.number_of_nodes()
    
    closed_hoods = [[i] + list(G.neighbors(i)) for i in range(N)]
    initial_degrees = [G.degree(i) for i in range(N)]
    
    vals = [rng.random() for _ in range(N)]
    
    max_deg_node = max(range(N), key=lambda i: initial_degrees[i])
    max_d = initial_degrees[max_deg_node]
    deg1_nodes = [i for i, d in enumerate(initial_degrees) if d == 1]
    
    global_exps = [sum(vals) / N]
    hub_exps = [vals[max_deg_node]]
    deg1_exps = [sum(vals[i] for i in deg1_nodes) / len(deg1_nodes) if deg1_nodes else 0.0]
    
    history_minus_1 = None
    history_minus_2 = vals.copy()
    
    step = 0
    final_vals = None
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
            global_exps.append(sum(vals) / N)
            hub_exps.append(vals[max_deg_node])
            deg1_exps.append(sum(vals[i] for i in deg1_nodes) / len(deg1_nodes) if deg1_nodes else 0.0)
            
        if not changed:
            final_vals = new_vals
            break
            
        if history_minus_2 is not None and new_vals == history_minus_2:
            final_vals = history_minus_1
            break
            
        if step > 150:
            final_vals = vals
            break

    while len(global_exps) <= max_t:
        global_exps.append(sum(final_vals) / N)
        hub_exps.append(final_vals[max_deg_node])
        deg1_exps.append(sum(final_vals[i] for i in deg1_nodes) / len(deg1_nodes) if deg1_nodes else 0.0)

    return global_exps, hub_exps, deg1_exps, max_d

# =============================================================================
# PART 4: MASTER SIMULATION & IMMEDIATE PLOTTING
# =============================================================================

def plot_single_tau_trajectory(tau, g_traj, h_traj, d1_traj, avg_dmax, max_t, runs):
    """Calculates bounds, prints results, and plots the trajectory for a single tau."""
    print(f"\nCalculating theoretical bounds for \u03C4 = {tau}...")
    analyzer = MaxLpaBounds(tau)
    
    g_t1 = analyzer.compute_t1_expectation()
    g_tinf = analyzer.compute_general_bounds()
    h_t1 = (avg_dmax + 1) / (avg_dmax + 2)
    h_tinf = 1/2
    d1_t1 = 2/3
    d1_tinf = analyzer.bound_degree_1()
    
    b_global = [0.5, g_t1] + [g_tinf] * (max_t - 1)
    b_hub = [0.5, h_t1] + [h_tinf] * (max_t - 1)
    b_deg1 = [0.5, d1_t1] + [d1_tinf] * (max_t - 1)
    
    bnds = {'global': b_global, 'hub': b_hub, 'deg1': b_deg1}
    
    g_means = np.mean(g_traj, axis=0)
    h_means = np.mean(h_traj, axis=0)
    d1_means = np.mean(d1_traj, axis=0)
    
    def get_moe(traj):
        if runs > 1: return 1.96 * (np.std(traj, axis=0, ddof=1) / np.sqrt(runs))
        return np.zeros_like(g_means)
        
    g_moe = get_moe(g_traj)
    h_moe = get_moe(h_traj)
    d1_moe = get_moe(d1_traj)
    t_vals = np.arange(max_t + 1)
    
    print("\n" + "="*80)
    print(f" EXPECTED VALUES vs THEORETICAL BOUNDS (\u03C4 = {tau})")
    print("="*80)
    for t_idx in t_vals:
        print(f"  t={t_idx}:")
        print(f"    Hub Node:   {h_means[t_idx]:.4f} ± {h_moe[t_idx]:.4f}  (Bound: >= {bnds['hub'][t_idx]:.4f})")
        print(f"    Global Avg: {g_means[t_idx]:.4f} ± {g_moe[t_idx]:.4f}  (Bound: >= {bnds['global'][t_idx]:.4f})")
        print(f"    Degree-1:   {d1_means[t_idx]:.4f} ± {d1_moe[t_idx]:.4f}  (Bound: >= {bnds['deg1'][t_idx]:.4f})")
    print("="*80 + "\n")
    
    plt.figure(figsize=(9, 6))
    plt.plot(t_vals, h_means, marker='^', linestyle='-', color='firebrick', markersize=8, linewidth=2, label="$d_{\\text{max}}$ (Simulated)")
    plt.plot(t_vals, g_means, marker='s', linestyle='-', color='black', markersize=8, linewidth=2.5, label="Global (Simulated)")
    plt.plot(t_vals, d1_means, marker='v', linestyle='-', color='steelblue', markersize=8, linewidth=2, label="Leaf (Simulated)")
    plt.plot(t_vals, bnds['global'], linestyle='--', color='black', alpha=0.6, linewidth=2.5, label="Global Lower Bound")
    plt.plot(t_vals, bnds['deg1'], linestyle='--', color='steelblue', alpha=0.6, linewidth=2, label="Leaf Lower Bound")

    plt.title(f"Expected Label Values vs Theoretical Bounds ($\u03C4={tau}$)", fontsize=14)
    plt.xlabel("Time Step (t)", fontsize=12)
    plt.ylabel(r"Expected Label Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(t_vals)
    plt.legend(loc='lower right', fontsize=10, ncol=2)
    plt.tight_layout()
    plt.show()
    
    return bnds

def simulate_and_plot_master(num_nodes=3000, tau_values=[2.1, 2.5, 2.9], runs=50, max_t=5):
    start_time = time.time()
    rng = np.random.default_rng()
    
    global_exp_data = {tau: np.zeros((runs, max_t + 1)) for tau in tau_values}
    hub_exp_data = {tau: np.zeros((runs, max_t + 1)) for tau in tau_values}
    deg1_exp_data = {tau: np.zeros((runs, max_t + 1)) for tau in tau_values}
    avg_dmax_data = {tau: 0.0 for tau in tau_values}
    theoretical_bounds = {}

    print(f"Starting expected values simulation on ENTIRE GRAPH (N={num_nodes}, Runs={runs})...")
    
    for tau in tau_values:
        print(f"\nSimulating tau = {tau}...")
        dmax_list = []
        for run_idx in tqdm(range(runs), desc=f"Networks", leave=False):
            G = generate_scale_free_graph(num_nodes, tau)
            g_exps, h_exps, d1_exps, max_d = run_unified_max_lpa_fast_expected(G, rng, max_t)
            
            global_exp_data[tau][run_idx, :] = g_exps
            hub_exp_data[tau][run_idx, :] = h_exps
            deg1_exp_data[tau][run_idx, :] = d1_exps
            dmax_list.append(max_d)
                
        avg_dmax_data[tau] = np.mean(dmax_list)
        
        # Plot and print specifically for this tau immediately
        bnds = plot_single_tau_trajectory(
            tau, global_exp_data[tau], hub_exp_data[tau], 
            deg1_exp_data[tau], avg_dmax_data[tau], max_t, runs
        )
        theoretical_bounds[tau] = bnds

    print(f"\nSimulation phase complete in {time.time() - start_time:.2f} seconds.")
    
    # --- Final Tau Sweep Plot ---
    print("\nGenerating final Converged Expected Label Value vs Scale-Free Exponent sweep...")
    sorted_taus = sorted(list(tau_values))
    final_g_means, final_g_moes = [], []
    final_h_means, final_h_moes = [], []
    final_d1_means, final_d1_moes = [], []
    tinf_g_bounds, tinf_h_bounds, tinf_d1_bounds = [], [], []
    
    for tau in sorted_taus:
        g_traj = global_exp_data[tau][:, max_t]
        h_traj = hub_exp_data[tau][:, max_t]
        d1_traj = deg1_exp_data[tau][:, max_t]
        
        final_g_means.append(np.mean(g_traj))
        final_h_means.append(np.mean(h_traj))
        final_d1_means.append(np.mean(d1_traj))
        
        tinf_g_bounds.append(theoretical_bounds[tau]['global'][-1])
        tinf_h_bounds.append(theoretical_bounds[tau]['hub'][-1])
        tinf_d1_bounds.append(theoretical_bounds[tau]['deg1'][-1])
        
        if runs > 1:
            final_g_moes.append(1.96 * np.std(g_traj, ddof=1) / np.sqrt(runs))
            final_h_moes.append(1.96 * np.std(h_traj, ddof=1) / np.sqrt(runs))
            final_d1_moes.append(1.96 * np.std(d1_traj, ddof=1) / np.sqrt(runs))
        else:
            final_g_moes.append(0.0); final_h_moes.append(0.0); final_d1_moes.append(0.0)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_taus, final_h_means, marker='^', linestyle='-', color='firebrick', markersize=8, linewidth=2, label="$d_{\\text{max}}$ (Simulated)")
    plt.plot(sorted_taus, final_g_means, marker='s', linestyle='-', color='black', markersize=8, linewidth=2.5, label="Global (Simulated)")
    plt.plot(sorted_taus, final_d1_means, marker='v', linestyle='-', color='steelblue', markersize=8, linewidth=2, label="Leaf (Simulated)")
    plt.plot(sorted_taus, tinf_g_bounds, linestyle='--', color='black', alpha=0.6, linewidth=2.5, label="Global Lower Bound")
    plt.plot(sorted_taus, tinf_d1_bounds, linestyle='--', color='steelblue', alpha=0.6, linewidth=2, label="Leaf Lower Bound")

    plt.title(f"Converged Expected Label Value vs Scale-Free Exponent", fontsize=14)
    plt.xlabel(r"Scale-Free Exponent ($\tau$)", fontsize=12)
    plt.ylabel(r"Expected Label Value at Convergence", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sorted_taus)
    plt.legend(loc='lower left', fontsize=10, ncol=2)
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    simulate_and_plot_master(num_nodes=100000, tau_values=[2.1, 2.5, 2.9], runs=1000, max_t=5)