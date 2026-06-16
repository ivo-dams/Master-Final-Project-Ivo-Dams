
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:56:50 2026

@author: 20174931

Combined Script for k-ary Tree Mechanisms and Expectations (t=2)
Provides Figures 4.2, 4.3, and 4.4.

This script calculates the exact symbolic polynomials for the mutually exclusive 
survival mechanisms of MAX-LPA at t=2 for both middle and leaf-parent vertices. 
It uses SymPy to:
1. Evaluate the probability breakdown of mechanisms at eps=1 (bar charts).
2. Calculate the exact expected label value by integrating the PDF (line chart).
"""

import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_middle_polynomials(k):
    """
    Constructs the exact probability polynomials for an internal middle vertex at t=2.
    Aligns directly with Lemmas 4.6 to 4.10.
    """
    x, y, eps = sp.symbols('x y eps')

    # Lemma 4.6: Bond initiated by v
    poly_bond1 = sp.integrate(x**(k+1) * (1-(1-x**k)**(k+1)), (x, 0, eps))
    
    # Lemma 4.7: Bond initiated by a neighbour
    B_sum = sum(sp.binomial(k, m) * (x**(k+1))**m * (y - x**(k+1))**(k-m) for m in range(min(3, k+1)))
    poly_bond2 = (k+1) * sp.integrate(sp.integrate(y**k * B_sum, (y, x, eps)), (x, 0, eps))

    # Lemma 4.8: Standard Flip-flop (j=2)
    if k < 2:
        poly_ff1 = sp.S.Zero
    else:
        intruder_logic = sp.integrate((k-1) * (y-x**(k+1))**(k-2) * (1-y**k), (y, x, 1))
        poly_ff1 = sp.binomial(k + 1, 2) * sp.integrate(x**(2*(k+1)) * intruder_logic, (x, 0, eps))

    # Lemma 4.9: Super Flip-flop (j>=3)
    poly_ff2 = sp.S.Zero
    for m in range(3, k+1):
        term = sp.binomial(k+1, m) * sp.integrate(
            x**(m*(k+1)) * ((1-x**(k+1))**(k+1-m) - (x-x**(k+1))**(k+1-m)), 
            (x, 0, eps)
        )
        poly_ff2 += term
    
    # Lemma 4.10: Residual State (No Duplicates)
    ball_size = k**2 + 2*k + 2
    prob_nodups = 1 - poly_bond1.subs(eps, 1) - poly_bond2.subs(eps, 1) - poly_ff1.subs(eps, 1) - poly_ff2.subs(eps, 1)
    poly_nodups = prob_nodups * eps**ball_size

    poly_total = poly_bond1 + poly_bond2 + poly_ff1 + poly_ff2 + poly_nodups

    return {
        'bond1': poly_bond1,
        'bond2': poly_bond2,
        'ff1': poly_ff1,
        'ff2': poly_ff2,
        'nodups': poly_nodups,
        'total': poly_total
    }

def get_leaf_parent_polynomials(k):
    """
    Constructs the exact probability polynomials for a leaf-parent vertex at t=2.
    Aligns directly with Lemmas 4.12 to 4.16.
    """
    x, y, eps = sp.symbols('x y eps')
    
    # Lemma 4.12: Bond initiated by v
    poly_bond1 = (1/(k+2)) * eps**(k+2)
    
    # Lemma 4.13: Bond initiated by a neighbour (Parent + Leaf configurations)
    B_sum = sum(sp.binomial(k, j) * x**j * (y-x)**(k-j) for j in range(min(3, k+1)))
    poly_bond2 = sp.integrate(sp.integrate(y**k * B_sum, (y, x, eps)), (x, 0, eps))
    
    term1 = y*(y-x)**(k-1)
    term2 = (k-1)*x*y*(y-x)**(k-2) if k >= 2 else 0
    term3 = sp.binomial(k-1, 2)*x**2*(y-x**(k+1))*(y-x)**(k-3) if k >= 3 else 0
    leaves = term1 + term2 + term3
    poly_bond2 += k * sp.integrate(sp.integrate(leaves, (y, x, eps)), (x, 0, eps))

    # Lemma 4.14: Standard Flip-flop (j=2)
    poly_ff1 = sp.S.Zero
    if k >= 2:
        intruder_logic = sp.integrate((y-x)**(k-2) * (1-y**k), (y, x, 1))
        poly_ff1 = sp.binomial(k, 2) * sp.integrate(x**2 * intruder_logic, (x, 0, eps))
        
    # Lemma 4.15: Super Flip-flop (j>=3)
    poly_ff2 = sp.S.Zero
    if k >= 3:
        term1_ff2 = sp.binomial(k, 2) * sp.integrate(x**(k+3) * (1-x)**(k-2), (x, 0, eps))
        term2_ff2 = sum(sp.binomial(k,j) * sp.integrate(x**j * (1-x)**(k-j), (x, 0, eps)) for j in range(3, k))
        term3_ff2 = sp.integrate(x**k * (1-x), (x, 0, eps))
        poly_ff2 = term1_ff2 + term2_ff2 + term3_ff2
        
    # Lemma 4.16: Residual State (No Duplicates)
    ball_size = 2*k + 2
    prob_nodups = 1 - poly_bond1.subs(eps, 1) - poly_bond2.subs(eps, 1) - poly_ff1.subs(eps, 1) - poly_ff2.subs(eps, 1)
    poly_nodups = prob_nodups * eps**ball_size

    poly_total = poly_bond1 + poly_bond2 + poly_ff1 + poly_ff2 + poly_nodups

    return {
        'bond1': poly_bond1,
        'bond2': poly_bond2,
        'ff1': poly_ff1,
        'ff2': poly_ff2,
        'nodups': poly_nodups,
        'total': poly_total
    }

def get_leaf_polynomial(k):
    """CDF for a leaf vertex at t=2 (Lemma 4.5)"""
    eps = sp.symbols('eps')
    return eps**(k+2)

def expected_value(poly_total):
    """Derives the PDF from the total CDF and computes the expected label value."""
    eps = sp.symbols('eps')
    pdf = sp.diff(poly_total, eps)
    return float(sp.integrate(eps * pdf, (eps, 0, 1)))

def plot_mechanisms(data, title, filename):
    """Helper function to cleanly plot the stacked bar charts for survival mechanisms."""
    ks = [d['k'] for d in data]
    categories = [
        'Bond initiated by v',
        'Bond initiated by neighbor of v',
        'Flip-flop',
        'Super flip-flop',
        'No duplicates'
    ]
    
    plt.figure(figsize=(12, 7))
    bottom = [0] * len(ks)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, category in enumerate(categories):
        values = [d[category] for d in data]
        plt.bar(ks, values, bottom=bottom, label=category, color=colors[i])
        bottom = [bottom[j] + values[j] for j in range(len(ks))]
        
    plt.xlabel('Branching Factor (k)', fontsize=14)
    plt.ylabel('Probability of Event', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(ks)
    
    # CORRECTED: Placed the legend exactly as it was in the original scripts
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to accommodate the external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def main(k_values):
    
    # Data storage
    middle_probs = []
    leaf_parent_probs = []
    
    exp_middle = []
    exp_leaf_parent = []
    exp_leaf = []
    exp_infinity = []
    
    print("Computing symbolic expectations and mechanisms for k=1 to 20...")
    for k in tqdm(k_values, desc="Processing k values"):
        eps = sp.symbols('eps')
        
        # --- 1. Middle Vertex (Layers L_2 to L_{h-2}) ---
        mid_polys = get_middle_polynomials(k)
        middle_probs.append({
            'k': k,
            'Bond initiated by v': float(mid_polys['bond1'].subs(eps, 1)),
            'Bond initiated by neighbor of v': float(mid_polys['bond2'].subs(eps, 1)),
            'Flip-flop': float(mid_polys['ff1'].subs(eps, 1)),
            'Super flip-flop': float(mid_polys['ff2'].subs(eps, 1)),
            'No duplicates': float(mid_polys['nodups'].subs(eps, 1))
        })
        e_mid = expected_value(mid_polys['total'])
        exp_middle.append(e_mid)
        
        # --- 2. Leaf-Parent Vertex (Layer L_{h-1}) ---
        lp_polys = get_leaf_parent_polynomials(k)
        leaf_parent_probs.append({
            'k': k,
            'Bond initiated by v': float(lp_polys['bond1'].subs(eps, 1)),
            'Bond initiated by neighbor of v': float(lp_polys['bond2'].subs(eps, 1)),
            'Flip-flop': float(lp_polys['ff1'].subs(eps, 1)),
            'Super flip-flop': float(lp_polys['ff2'].subs(eps, 1)),
            'No duplicates': float(lp_polys['nodups'].subs(eps, 1))
        })
        e_lp = expected_value(lp_polys['total'])
        exp_leaf_parent.append(e_lp)
        
        # --- 3. Leaf Vertex (Layer L_h) ---
        poly_leaf = get_leaf_polynomial(k)
        e_l = expected_value(poly_leaf)
        exp_leaf.append(e_l)
        
        # --- 4. Global Expectation (Theorem 4.18) ---
        if k == 1:
            e_inf = e_mid 
        else:
            e_inf = (1 - 1/k) * e_l + (1/k - 1/(k**2)) * e_lp + (1/(k**2)) * e_mid
        exp_infinity.append(e_inf)

    # --- Plotting Figure 4.3: Middle Vertex Mechanisms ---
    plot_mechanisms(middle_probs, 'Middle vertex: Probability Breakdown for k-ary Mechanisms', 'middle_mechanisms.png')
    
    # --- Plotting Figure 4.4: Leaf-Parent Mechanisms ---
    plot_mechanisms(leaf_parent_probs, 'Leaf-Parent Vertex: Probability Breakdown for k-ary Mechanisms', 'leaf_parent_mechanisms.png')
    
    # --- Plotting Figure 4.2: Expected Label Values ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, exp_leaf, marker='^', linewidth=2, label="Leaf Vertex")
    plt.plot(k_values, exp_leaf_parent, marker='s', linewidth=2, label="Leaf-Parent Vertex")
    plt.plot(k_values, exp_middle, marker='o', linewidth=2, label="Middle Vertex")
    
    plt.plot(k_values, exp_infinity, marker='*', markersize=10, linewidth=2, linestyle='--', color='purple', label=r"Any Random Vertex ($n \to \infty$)")
    
    plt.xlabel('Branching Factor (k)', fontsize=14)
    plt.ylabel('Expected Label Value', fontsize=14)
    plt.title('Expected Label Values at t=2', fontsize=16)
    plt.xticks(k_values)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

k_values = list(range(1, 21))
if __name__ == "__main__":
    main(k_values)