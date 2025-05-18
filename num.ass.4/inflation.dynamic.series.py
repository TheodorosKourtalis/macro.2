#!/usr/bin/env python3
"""
inflation.dynamic.series.py

1) Simulate a deterministic Phillips‐curve experiment over T=20 periods  
2) Plot:
   • Left panel: step‐function inflation path with policy shift marker  
   • Right panel: step‐function unemployment under constant, adaptive, and rational expectations  
3) Add a centered black citation at the bottom  
4) Save the figure as “phillips_step_plot.png” and print its saved location
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1) Style
    sns.set_style("whitegrid")

    # 2) Parameters
    alpha = 1.5           # Phillips‐curve slope
    u_n   = 0.05          # natural rate of unemployment
    T     = 20            # number of periods

    # 3) Simulate inflation
    pi  = np.zeros(T+1)   # π[0] is π_{-1}
    mu  = np.zeros(T+1)
    for t in range(1, T+1):
        mu[t] = 0.0 if t < 10 else 0.005
        pi[t] = mu[t] + pi[t-1]
    pi = pi[1:]           # now pi[i] == π_{i+1}

    # 4) Expectations
    pi_e_const    = np.zeros_like(pi)                       # constant
    pi_e_adaptive = np.concatenate(([0.0], pi[:-1]))        # adaptive (lagged π)
    pi_lag        = np.concatenate(([0.0], pi[:-1]))
    pi_e_rational = mu[1:] + pi_lag                         # fully rational

    # 5) Unemployment
    u_const    = u_n + (pi_e_const    - pi) / alpha
    u_adaptive = u_n + (pi_e_adaptive - pi) / alpha
    u_rat      = u_n + (pi_e_rational - pi) / alpha

    # 6) Build manual step‐arrays for adaptive so jumps hit exactly at integer t
    x_adapt = np.empty(2*T)
    y_adapt = np.empty(2*T)
    for i in range(T):
        x_adapt[2*i]   = i+1
        x_adapt[2*i+1] = i+2
        y_adapt[2*i]   = u_adaptive[i]
        y_adapt[2*i+1] = u_adaptive[i]

    # 7) Plot
    periods = np.arange(1, T+1)
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Inflation panel (true inflation in black)
    axes[0].step(periods, pi, where='post', color='black', lw=2, label=r'$\pi_t$')
    axes[0].plot([], [], color='red', ls=':', label="Policy shift at $t=10$")  # dummy for legend
    axes[0].axvline(10, color='red', ls=':')
    axes[0].set(title="Inflation Path", xlabel="Period $t$", ylabel=r"$\pi_t$")
    axes[0].legend()

    # Unemployment panel
    axes[1].step(periods, u_const,     where='post', lw=2, label="Constant exp.")
    axes[1].plot(x_adapt, y_adapt,     lw=2, color='grey', label="Adaptive exp.")
    axes[1].step(periods, u_rat,       where='post', color='green', lw=2, label="Rational exp.")
    axes[1].axvline(10, color='red', ls=':')
    axes[1].set(title="Unemployment under Different Expectations",
                xlabel="Period $t$", ylabel=r"$u_t$")
    axes[1].legend()

    # 8) Add centered citation at bottom
    citation = "Calculations by Theodoros Kourtalis for MT2 – 1412"
    fig.text(0.5, 0.01, citation, ha='center', va='bottom',
             fontsize=9, color='black')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # 9) Save figure and print its path
    fname = "phillips_step_plot.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {os.path.abspath(fname)}")

    plt.show()

if __name__ == "__main__":
    main()