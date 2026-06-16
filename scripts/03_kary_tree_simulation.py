# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:20:44 2026

@author: 20174931

Finalised k-ary Tree Simulation Script (k >= 3)

Provides Data and Figures for General Branching Factors (e.g., Figures 4.12, 4.13, 6.4).

This script implements:
1. Exact theoretical expected bounds utilizing SymPy constants.
2. Fast stratified global simulation mapping proportional layer weights.
3. Isolated middle-vertex simulation for internal convergence tracking.
4. Active-set full tree simulation to calculate Lonely Vertex probabilities 
   across varying branching factors (k).
"""

import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sympy as sp
import math

# =============================================================================
# PART 1: EXACT SYMBOLIC CONSTANTS
# =============================================================================

def compute_k_ary_exact_values(k):
    """
    Computes exact probabilities and partial expectations for the survival 
    mechanisms of MAX-LPA at t=2 using SymPy symbolic integrations.
    
    This function mathematically formalises the formation of bonds, flip-flops, 
    and residual states (no duplicates) on acyclic complete k-ary trees.
    """
    x, y, eps = sp.symbols('x y eps')

    # --- Middle Node Constants (Vertices isolated from the leaf boundary) ---
    
    # Probability of an unbreakable bond initiated by the vertex v itself
    poly_bond1 = sp.integrate(x**(k+1) * (1-(1-x**k)**(k+1)), (x, 0, eps))
    
    # Probability of an unbreakable bond initiated by a neighbor of v
    B_sum = sum(sp.binomial(k,m) * ((x**(k+1))**m) * (y - x**(k+1))**(k-m) for m in range(0, 3))
    poly_bond2 = (k+1) * sp.integrate(sp.integrate(y**k * B_sum, (y, x, eps)), (x, 0, eps))
    poly_bond = poly_bond1 + poly_bond2
    
    # Standard Flip-flops (exactly j=2 identical highest labels)
    intruder_logic = sp.integrate((k-1) * (y-x**(k+1))**(k-2) * (1-y**k), (y, x, 1))
    poly_ff1 = sp.binomial(k + 1, 2) * sp.integrate(x**(2*(k+1)) * intruder_logic, (x, 0, eps))
    
    # Super Flip-flops (j >= 3 identical highest labels)
    ff_m_probs = {}
    poly_ff2 = sp.S.Zero
    for m in range(3, k+1):
        term = sp.binomial(k+1, m) * sp.integrate(x**(m*(k+1)) * ((1-x**(k+1))**(k+1-m) - (x-x**(k+1))**(k+1-m)), (x, 0, eps))
        poly_ff2 += term
        ff_m_probs[m] = float(term.subs(eps, 1))

    # Residual State: The probability that no duplicates exist to form the above mechanisms
    prob_nodups = 1 - poly_bond.subs(eps, 1) - poly_ff1.subs(eps, 1) - poly_ff2.subs(eps, 1)
    
    # Safe Bond calculation (Strictly unbreakable bounds initiated by v)
    P_safe_sum = sum(sp.binomial(k, j) * y**(j*(k+1)) * (x - y**(k+1))**(k-j) for j in range(0, 3))
    P_safe = (1/x) * sp.integrate(P_safe_sum, (y, 0, x))
    poly_b_init_safe = sp.integrate(x**(k+1) * (1 - (1 - P_safe)**(k+1)), (x, 0, eps))
    
    # Compute partial expectations by integrating the Probability Density Functions (PDF)
    pdf_b_init_safe = sp.diff(poly_b_init_safe, eps)
    E_b_init_safe_partial = float(sp.integrate(eps * pdf_b_init_safe, (eps, 0, 1)))
    
    pdf_b_join = sp.diff(poly_bond2, eps)
    E_b_join_partial = float(sp.integrate(eps * pdf_b_join, (eps, 0, 1)))
    
    P_b_init = float(poly_bond1.subs(eps, 1))
    P_b_init_safe = float(poly_b_init_safe.subs(eps, 1))
    P_b_join = float(poly_bond2.subs(eps, 1))
    
    # Total combined expectation at t=2 for a middle vertex
    pdf_total = sp.diff(poly_bond + poly_ff1 + poly_ff2 + (prob_nodups * eps**(k**2 + 2*k + 2)), eps)
    E_mid_t2 = float(sp.integrate(eps * pdf_total, (eps, 0, 1)))

    # --- Leaf-Parent (t=2) Exact Expectation ---
    # Calculates mechanisms for a vertex exactly one layer above the leaves
    poly_l1_bond1 = 1/(k+2) * eps**(k+2)
    B_sum_l1 = sum(sp.binomial(k,j) * x**j * (y-x)**(k-j) for j in range(0, 3))
    poly_l1_bond2 = sp.integrate(sp.integrate(y**k * B_sum_l1, (y, x, eps)), (x, 0, eps))
    leaves = y*(y-x)**(k-1) + (k-1)*x*y*(y-x)**(k-2) + sp.binomial(k-1,2)*x**2*(y-x**(k+1))*(y-x)**(k-3)
    poly_l1_bond2 += k * sp.integrate(sp.integrate(leaves, (y, x, eps)), (x, 0, eps))

    poly_l1_ff1 = sp.S.Zero
    if k >= 2:
        intruder_logic = sp.integrate((y-x)**(k-2) * (1-y**k), (y, x, 1))
        poly_l1_ff1 = sp.binomial(k, 2) * sp.integrate(x**2 * intruder_logic, (x, 0, eps))

    poly_l1_ff2 = sp.S.Zero
    if k >= 3:
        poly_l1_ff2 += sp.integrate(sp.binomial(k,2) * x**(k+3) * (1-x)**(k-2), (x, 0, eps))
        for j in range(3, k):
            poly_l1_ff2 += sp.binomial(k, j) * sp.integrate(x**j * (1-x)**(k-j), (x, 0, eps))
        poly_l1_ff2 += sp.integrate(x**k * (1-x), (x, 0, eps))
    
    prob_l1_nd = 1 - poly_l1_bond1.subs(eps, 1) - poly_l1_bond2.subs(eps, 1) - poly_l1_ff1.subs(eps, 1) - poly_l1_ff2.subs(eps, 1)
    poly_l1_total = poly_l1_bond1 + poly_l1_bond2 + poly_l1_ff1 + poly_l1_ff2 + (prob_l1_nd * eps**(2*k + 2))
    pdf_l1 = sp.diff(poly_l1_total, eps)
    E_l1_t2 = float(sp.integrate(eps * pdf_l1, (eps, 0, 1)))

    # --- Infinite Time Limit (Asymptotic Expectation) ---
    c1 = 1 - (2 / ((k + 1) * (k + 2)))
    c2 = 2 / ((k + 1) * (k + 2))
    E_inf = 1 - (c1 / (k + 3) + c2 / (2 * k + 3))

    return {
        'E_b_init_safe_partial': E_b_init_safe_partial,
        'E_b_join_partial': E_b_join_partial,
        'P_b_init': P_b_init,
        'P_b_init_safe': P_b_init_safe,
        'P_b_join': P_b_join,
        'P_FF2': float(poly_ff1.subs(eps, 1)),
        'P_FF_m': ff_m_probs,
        'P_FF_all': float(poly_ff1.subs(eps, 1) + poly_ff2.subs(eps, 1)),
        'P_ND': float(prob_nodups),
        'E_mid_t2': E_mid_t2,
        'E_l1_t2': E_l1_t2,
        'E_inf': E_inf
    }


# =============================================================================
# PART 2: THEORETICAL BOUNDS LOGIC
# =============================================================================

def get_n_dt(k, t, d):
    size = 0
    h_v = min(t, d)
    size += (k**(h_v + 1) - 1) // (k - 1)
    size += t
    for i in range(1, t + 1):
        h_sib = min(t - i - 1, d + i - 1)
        if h_sib >= 0:
            # Multiplying the single branch formula by (k-1) cancels the denominator.
            size += (k**(h_sib + 1) - 1) 
    return size

def theoretical_middle_bounds(k, max_t, consts):
    """
    Computes exact upper and lower analytical bounds exclusively for internal 
    middle vertices (vertices strictly isolated from leaf-boundary effects).
    """
    t_values = np.arange(1, max_t + 1)
    lower_bounds, upper_bounds = [], []
    
    # Calculate the fixed, structural t>=3 Lower Bound floor
    B2_size = k**2 + 2*k + 2
    denom_B2 = B2_size + 1
    
    # The unbreakable bond expectations set the permanent baseline
    LB_mid_t3 = consts['E_b_init_safe_partial'] + consts['E_b_join_partial']
    LB_mid_t3 += (consts['P_b_init'] - consts['P_b_init_safe']) * ((2*k + 2) / denom_B2)
    LB_mid_t3 += consts['P_FF2'] * ((2*k + 3) / denom_B2)
    for m, prob in consts['P_FF_m'].items():
        LB_mid_t3 += prob * ((m*(k+1) + 1) / denom_B2)
    LB_mid_t3 += consts['P_ND'] * ((k + 2) / (k + 3))
    
    # Probability mass of configurations that are NOT permanently stabilised
    unstable_prob = 1 - consts['P_b_init_safe'] - consts['P_b_join']
    
    for t in t_values:
        if t == 1: 
            val = (k + 2) / (k + 3)
            lower_bounds.append(val)
            upper_bounds.append(val)
        elif t == 2: 
            val = consts['E_mid_t2']
            lower_bounds.append(val)
            upper_bounds.append(val)
        else:
            n_mid = get_n_dt(k, t, t)
            LB_mid = LB_mid_t3
            
            # The upper bound assumes unstable probability mass maximizes its potential
            UB_mid = consts['E_b_init_safe_partial'] + consts['E_b_join_partial'] + unstable_prob * (n_mid / (n_mid + 1))
            lower_bounds.append(LB_mid)
            upper_bounds.append(UB_mid)
            
    return t_values, lower_bounds, upper_bounds

def theoretical_global_bounds(k, max_t, consts):
    """
    Computes exact fractional analytical bounds for the entire global k-ary tree.
    Aggregates the individual bounds of each depth layer weighted by the 
    fraction of the population existing at that depth.
    """
    t_values = np.arange(1, max_t + 1)
    lower_bounds, upper_bounds = [], []
    
    B2_size = k**2 + 2*k + 2
    denom_B2 = B2_size + 1
    
    LB_mid_t3 = consts['E_b_init_safe_partial'] + consts['E_b_join_partial']
    LB_mid_t3 += (consts['P_b_init'] - consts['P_b_init_safe']) * ((2*k + 2) / denom_B2)
    LB_mid_t3 += consts['P_FF2'] * ((2*k + 3) / denom_B2)
    for m, prob in consts['P_FF_m'].items():
        LB_mid_t3 += prob * ((m*(k+1) + 1) / denom_B2)
    LB_mid_t3 += consts['P_ND'] * ((k + 2) / (k + 3))
    
    unstable_prob = 1 - consts['P_b_init_safe'] - consts['P_b_join']
    
    for t in t_values:
        LB_total, UB_total = 0, 0
        
        # Iterate over layers close enough to the leaves to be bounded by them (d < t)
        for d in range(t):
            frac_d = (k - 1) / (k ** (d + 1)) # Fraction of total nodes in layer d
            
            if d == 0: # Leaf boundary exact values
                if t == 1: val = 2 / 3
                elif t == 2: val = (k + 2) / (k + 3)
                else: val = consts['E_inf']
                LB_d = val; UB_d = val
            elif d == 1: # Leaf-parent exact values and dynamically expanding ceilings
                if t == 1: 
                    LB_d = (k + 2) / (k + 3)
                    UB_d = (k + 2) / (k + 3)
                elif t == 2: 
                    LB_d = consts['E_l1_t2']
                    UB_d = consts['E_l1_t2']
                else: 
                    # The lower bound is the strict t=1 topological floor. 
                    LB_d = (k + 2) / (k + 3)
                    # The upper bound is the expanding maximum order statistic of B_t.
                    n_dt = get_n_dt(k, t, d=1)
                    UB_d = n_dt / (n_dt + 1)
            else: # Internal layers that truncate due to the boundary
                n_dt = get_n_dt(k, t, d)
                LB_d = LB_mid_t3
                UB_d = consts['E_b_init_safe_partial'] + consts['E_b_join_partial'] + unstable_prob * (n_dt / (n_dt + 1))
            
            LB_total += frac_d * LB_d
            UB_total += frac_d * UB_d
            
        # The remainder of the tree acts as a pure middle vertex
        frac_mid = 1 / (k ** t)
        if t == 1: 
            val_mid = (k + 2) / (k + 3)
            LB_mid = val_mid; UB_mid = val_mid
        elif t == 2: 
            val_mid = consts['E_mid_t2']
            LB_mid = val_mid; UB_mid = val_mid
        else:
            n_mid = get_n_dt(k, t, t)
            LB_mid = LB_mid_t3
            UB_mid = consts['E_b_init_safe_partial'] + consts['E_b_join_partial'] + unstable_prob * (n_mid / (n_mid + 1))
            
        LB_total += frac_mid * LB_mid
        UB_total += frac_mid * UB_mid
        
        lower_bounds.append(LB_total)
        upper_bounds.append(UB_total)
        
    return t_values, lower_bounds, upper_bounds


# =============================================================================
# PART 3: FAST LOCALIZED SIMULATION (PROPORTIONAL STRATIFICATION)
# =============================================================================

def build_local_neighborhood(k, t, d):
    """
    Generates a localized NetworkX acyclic graph representing only the 
    distance-t closed neighborhood of a vertex at depth d.
    """
    G = nx.Graph()
    center = 0
    G.add_node(center, dist=0, h=d)
    
    node_counter = 1
    queue = [(center, d, 'center', 0)]
    
    # BFS generation to populate the local tree accurately
    while queue:
        curr, h, n_type, dist = queue.pop(0)
        
        if dist < t:
            num_up, num_down = 0, 0
            if n_type == 'center':
                num_up = 1; num_down = k
            elif n_type == 'ancestor':
                num_up = 1; num_down = k - 1
            elif n_type == 'descendant':
                num_up = 0; num_down = k
                
            # If h==0, we hit the leaf boundary; no further descendants exist
            if h == 0: num_down = 0 
                
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

def simulate_single_node_type(k, max_t, d, runs_d):
    """
    Simulates MAX-LPA expectations strictly for a specific class of node 
    (identified by its topological distance d from the leaves).
    """
    G, center = build_local_neighborhood(k, max_t, d)
    nodes = list(G.nodes())
    
    # Pre-sorting nodes by distance avoids computing redundant states
    nodes_by_dist = {dist: [] for dist in range(max_t + 1)}
    for n in nodes:
        nodes_by_dist[G.nodes[n]['dist']].append(n)
        
    closed_hoods = {n: [n] + list(G.neighbors(n)) for n in nodes}
    trajectories = np.zeros((runs_d, max_t))
    rng = np.random.default_rng()
    
    for run_idx in tqdm(range(runs_d), desc=f"Simulating layer d={d} ({runs_d} runs)", leave=False):
        vals = {node: rng.random() for node in nodes}
        
        # Iterative update loop for MAX-LPA
        for step in range(1, max_t + 1):
            new_vals = {}
            # Active optimization: Only compute for nodes capable of reaching the center in time
            max_dist_needed = max_t - step
            active_nodes = []
            for dist in range(max_dist_needed + 1):
                active_nodes.extend(nodes_by_dist[dist])
                
            if step == 1:
                for n in active_nodes:
                    new_vals[n] = max(vals[u] for u in closed_hoods[n])
            else:
                for n in active_nodes:
                    hood_vals = [vals[u] for u in closed_hoods[n]]
                    counts = Counter(hood_vals)
                    max_freq = max(counts.values())
                    new_vals[n] = max([v for v, c in counts.items() if c == max_freq])
            
            vals = new_vals
            trajectories[run_idx, step - 1] = vals[center]
            
    means = np.mean(trajectories, axis=0)
    variances = np.var(trajectories, axis=0, ddof=1) if runs_d > 1 else np.zeros(max_t)
    return means, variances

def fast_global_simulation(k=3, max_t=5, base_runs=10000):
    """
    Stratified global simulation: Scales simulation runs per depth exponentially,
    saving compute power by matching the mathematical volume of nodes per layer,
    and returns aggregated, proportional global expectations.
    """
    print(f"\nStarting Proportional Stratified Simulation (Base runs={base_runs})...")
    start_time = time.time()
    
    layer_stats = {}
    layer_runs = {}
    
    # Scale number of simulation runs exponentially based on population size in tree
    for d in range(max_t + 1):
        runs_d = max(2, base_runs // (k**d))
        layer_runs[d] = runs_d
        layer_stats[d] = simulate_single_node_type(k, max_t, d, runs_d)
        
    global_means = np.zeros(max_t)
    global_moe = np.zeros(max_t)
    
    # Reassemble the global expected value analytically from the independent stratum
    for step in range(max_t):
        t_idx = step
        t_current = step + 1
        expected_val = 0
        expected_var = 0
        
        for d in range(t_current):
            frac_d = (k - 1) / (k ** (d + 1))
            mean_d = layer_stats[d][0][t_idx]
            var_d = layer_stats[d][1][t_idx]
            runs_d = layer_runs[d]
            
            expected_val += frac_d * mean_d
            expected_var += (frac_d ** 2) * (var_d / runs_d)
            
        frac_mid = 1 / (k ** t_current)
        mean_mid = layer_stats[max_t][0][t_idx]
        var_mid = layer_stats[max_t][1][t_idx]
        runs_mid = layer_runs[max_t]
        
        expected_val += frac_mid * mean_mid
        expected_var += (frac_mid ** 2) * (var_mid / runs_mid)
        
        global_means[step] = expected_val
        moe = 1.96 * np.sqrt(expected_var) # 95% Confidence Interval calculation
        global_moe[step] = moe
        
    print(f"Fast Global Simulation completed in {time.time() - start_time:.4f} seconds.\n")
    return global_means, global_moe


# =============================================================================
# PART 4: MULTI-k LONELY VERTEX CONVERGENCE (FULL TREE)
# =============================================================================

def simulate_k_ary_probabilities(k_values, target_n=10000, runs=1000):
    """
    Active-set full tree simulation to compute Lonely Vertex probabilities
    across varying branching factors (k) as time approaches infinity.
    Dynamically scales Tree Height H to keep node count close to target_n.
    """
    rng = np.random.default_rng()
    k_probabilities = {}
    
    print(f"Starting Multi-k Simulation (Target Nodes ~ {target_n}, Runs={runs})\n")
    start_time = time.time()
    
    for k in k_values:
        # Dynamically calculate required Height (H) to approximate target_n nodes
        raw_h = math.log(target_n * (k - 1) + 1, k) - 1
        H = max(1, math.floor(raw_h)) 
        
        G = nx.balanced_tree(k, H)
        N = G.number_of_nodes()
        
        open_hoods = [list(G.neighbors(i)) for i in range(N)]
        closed_hoods = [open_hoods[i] + [i] for i in range(N)]
        total_lonely_vertices = 0
        
        print(f"Simulating k={k} (Calculated H={H}, Actual Nodes={N})...")
        
        for _ in tqdm(range(runs), desc=f"k={k} Convergence", leave=False):
            vals = rng.random(N).tolist()
            
            # Active set: At t=0, all nodes must be evaluated
            active_nodes = set(range(N))
            
            # Active set MAX-LPA implementation (Convergence driven)
            while active_nodes:
                new_vals = vals.copy() 
                next_active = set()
                
                for i in active_nodes:
                    counts = {}
                    max_freq = 0
                    best_val = -1.0
                    
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
                    
                    # If a node changes state, its entire closed neighborhood must be re-evaluated
                    if best_val != vals[i]:
                        next_active.update(closed_hoods[i])
                        
                vals = new_vals
                active_nodes = next_active
            
            # Global Lonely Check (Post-Convergence)
            # A vertex is lonely if it strictly dominates its neighborhood mathematically
            for i in range(N):
                my_val = vals[i]
                is_lonely = True
                
                for neighbor_idx in open_hoods[i]:
                    if my_val <= vals[neighbor_idx]:
                        is_lonely = False
                        break
                
                if is_lonely:
                    total_lonely_vertices += 1
                    
        # Determine the probability of any arbitrary node being lonely at t=infinity
        prob = total_lonely_vertices / (N * runs)
        k_probabilities[k] = prob
        print(f"  -> Found {total_lonely_vertices} isolated vertices. Prob = {prob:.6e}\n")
        
    print(f"Lonely Vertex Simulation complete in {time.time() - start_time:.2f}s.")
    return k_probabilities, target_n, runs


# =============================================================================
# PART 5: PLOTTING INTERFACES
# =============================================================================

def plot_fast_global_network(k=3, max_t=5, base_runs=10000):
    """Generates visual bounds vs simulation expectations for the global tree."""
    print(f"\n--- Generating Data for Global expected label value (k={k}) ---")
    consts = compute_k_ary_exact_values(k)
    t_values, lower_bounds, upper_bounds = theoretical_global_bounds(k, max_t, consts)
    sim_means, sim_moe = fast_global_simulation(k=k, max_t=max_t, base_runs=base_runs) 
    
    print(f"\n--- Global Tree Results (k={k}) ---")
    for step in range(max_t):
        t_val = step + 1
        mean_val = sim_means[step]
        moe_val = sim_moe[step]
        print(f"Global Expected value (t={t_val}): {mean_val:.6f}")
        print(f"95% Confidence Interval: [{mean_val - moe_val:.6f}, {mean_val + moe_val:.6f}] (+/- {moe_val:.6f})\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values[1:], lower_bounds[1:], 'b--', marker='o', label="Global Lower Bound", linewidth=2)
    plt.plot(t_values[1:], upper_bounds[1:], 'r--', marker='o', label="Global Upper Bound", linewidth=2)
    plt.fill_between(t_values[1:], lower_bounds[1:], upper_bounds[1:], color='gray', alpha=0.2, label="Bounded Region")
    
    plt.errorbar(t_values[1:], sim_means[1:], marker='s', linestyle='-', 
                 color='black', markersize=8, capsize=5, capthick=2, linewidth=2, 
                 label="Simulated Expectation")
    
    plt.title(f"Global Expected Label Value ($k={k}$)", fontsize=14)
    plt.xlabel("Time step ($t$)", fontsize=12)
    plt.ylabel(r"Global Expected Label Value", fontsize=12)
    plt.xticks(t_values[1:])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_middle_network_only(k=3, max_t=5, runs=10000):
    """Generates visual bounds vs simulation expectations for internal middle vertices."""
    print(f"\n--- Generating Data for Middle expected label value (k={k}) ---")
    consts = compute_k_ary_exact_values(k)
    t_values, lower_bounds, upper_bounds = theoretical_middle_bounds(k, max_t, consts)
    
    # Simulate single node type at distance max_t (so it acts dynamically as an internal middle vertex)
    sim_means, sim_vars = simulate_single_node_type(k, max_t, max_t, runs)
    sim_moe = 1.96 * np.sqrt(sim_vars / runs)
    
    print(f"\n--- Middle Vertex Results (k={k}) ---")
    for step in range(max_t):
        t_val = step + 1
        mean_val = sim_means[step]
        moe_val = sim_moe[step]
        print(f"Middle Expected value (t={t_val}): {mean_val:.6f}")
        print(f"95% Confidence Interval: [{mean_val - moe_val:.6f}, {mean_val + moe_val:.6f}] (+/- {moe_val:.6f})\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values[1:], lower_bounds[1:], 'b--', marker='o', label="Global Lower Bound", linewidth=2)
    plt.plot(t_values[1:], upper_bounds[1:], 'r--', marker='o', label="Global Upper Bound", linewidth=2)
    plt.fill_between(t_values[1:], lower_bounds[1:], upper_bounds[1:], color='gray', alpha=0.2, label="Bounded Region")
    
    plt.errorbar(t_values[1:], sim_means[1:], marker='s', linestyle='-', 
                 color='black', markersize=8, capsize=5, capthick=2, linewidth=2, 
                 label="Simulated Expectation")
    
    plt.title(f"Expected Label Value for a Middle Vertex ($k={k}$)", fontsize=14)
    plt.xlabel("Time step ($t$)", fontsize=12)
    plt.ylabel(r"Expected label value", fontsize=12)
    plt.xticks(t_values[1:])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_k_probabilities(k_probs, target_n, runs):
    """Plots the probability mass function of Lonely Vertices surviving at t=infinity."""
    k_vals = sorted(list(k_probs.keys()))
    probs = [k_probs[k] for k in k_vals]
    
    plt.figure(figsize=(9, 6))
    plt.bar([str(k) for k in k_vals], probs, color='darkorchid', edgecolor='black', alpha=0.8, width=0.6)
    
    plt.title("Lonely Vertex PMF by Branching Factor", fontsize=13)
    plt.xlabel("Branching Factor ($k$)", fontsize=11)
    plt.ylabel("Proportion of Vertices Remaining Lonely", fontsize=11)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    
    # Scientific formatting triggers for highly unlikely probabilities at large k
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Updated to run the targeted plots at k=5
    k_target = 5    
    max_t_target = 5
    
    # 1. Plot the global expectations (k=5)
    plot_fast_global_network(k=k_target, max_t=max_t_target, base_runs=10000000)
    
    # 2. Plot the middle expectations exclusively (k=5)
    plot_middle_network_only(k=k_target, max_t=max_t_target, runs=1000000)
    
    # 3. Simulate Lonely Vertices across multiple k
    print("\n--- Generating Multi-k Lonely Vertex Distribution ---")
    k_range = list(range(2, 11))  # Evaluates k=2 through k=10
    results, tgt_n, sim_runs = simulate_k_ary_probabilities(k_values=k_range, target_n=100000, runs=1000)
    plot_k_probabilities(results, tgt_n, sim_runs)