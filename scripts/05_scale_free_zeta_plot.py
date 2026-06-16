# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:57:09 2026

@author: 20174931
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import zeta

# Parameters
taus = np.linspace(2.01, 4, 50)   # tau > 2
k_max = 1_000_000                     # truncation of the infinite sum

def expression(tau, k_max):
    k = np.arange(1, k_max + 1)
    series = np.sum(1.0 / (k**tau * (k + 2)))
    return 1.0 - series / float(zeta(tau))

values = np.array([expression(tau, k_max) for tau in taus])

# -------------------------------------------------
# Plot 1: Linear tau scale
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(taus, values, marker='o')
plt.xlabel(r'$\tau$')
plt.ylabel(r'Expcted Label Value')
plt.title("Expected Label value in a Scale-Free Graph at $t=1$")
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------------------------
# Plot 2: Logarithmic distance from critical point
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(taus - 2, values, marker='o')
plt.xscale('log')
plt.xlabel(r'$\tau$ (log)')
plt.ylabel(r'Expcted Label Value')
plt.title("Expected Label value in a Scale-Free Graph at $t=1$")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
