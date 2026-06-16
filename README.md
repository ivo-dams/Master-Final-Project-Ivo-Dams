# Label propagation with MAX-LPA for Tree-Like Graphs.

This repository contains the simulation, mathematical bounds, and topological analysis code developed for the Master's thesis: Label propagation with MAX-LPA for Tree-Like Graphs (Department of Mathematics and Computer Science, Eindhoven University of Technology).

## Overview
This project mathematically models and computationally simulates the expected label values under the MAX-LPA (Maximum Label Propagation Algorithm) across various acyclic network topologies. 

The codebase provides exact analytical bounding using symbolic integration, alongside highly optimized active-set simulations to verify topological limits, track community formations and analyse the convergence time paradox.

### Key Features
* **Exact Analytical Bounds:** Utilizes `sympy` to symbolically calculate survival mechanisms (bonds, flip-flops, residual states) for $k$-ary trees.
* **Finite-Size Scaling:** Implements degree-distribution normalizations to align infinite mathematical bounds with finite simulated scale-free networks ($\tau \to 2$).
* **Optimized Active-Set Simulation:** Synchronous propagation engines capable of evaluating millions of nodes using targeted neighbourhood tracking and delayed cycle detection.
* **Convergence Paradox Analysis:** Evaluates the sub-linear convergence times of MAX-LPA on the Largest Connected Component (LCC) of scale-free graphs compared to highly branched trees.

## Repository Structure

All source code is located in the `scripts/` directory, logically ordered by topological complexity:

### 1. Binary and $k$-ary Trees
* `01_binary_tree_elv.py`: Exact bounds and global simulations for binary trees ($k=2$). Generates Expected Label Value (ELV) plots and tracks lonely vertices.
* `02_kary_tree_mechanisms.py`: Symbolically integrates the exact probabilities of mutually exclusive survival mechanisms (Figure 4.3 & 4.4).
* `03_kary_tree_simulation.py`: Fast stratified global simulation mapping proportional layer weights for $k \ge 3$.
* `04_kary_convergence_time.py`: Analyzes MAX-LPA convergence times against expanding tree heights.

### 2. Scale-Free Networks
* `05_scale_free_zeta_plot.py`: Generates baseline expected label plots for scale-free graphs at $t=1$.
* `06_scale_free_elv_no_lcc.py`: Simulates the entire scale-free graph (without LCC extraction) using finite-size theoretical normalization.
* `07_scale_free_communities_lcc.py`: Structural analysis on the LCC, calculating community PMF/CCDF and degree correlations.
* `08_scale_free_convergence_time.py`: Simulates convergence times on scale-free LCCs across varying power-law exponents.

## Installation and Requirements

To run the scripts and generate the plots locally, clone this repository and install the required dependencies.

```bash
git clone [https://github.com/yourusername/MAX-LPA-Acyclic-Networks.git](https://github.com/yourusername/MAX-LPA-Acyclic-Networks.git)
cd MAX-LPA-Acyclic-Networks
pip install -r requirements.txt
