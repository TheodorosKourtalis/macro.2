#!/usr/bin/env python3
"""
simulate_phillips_full_and_corr.py

1) Simulate 1,000 shocks εₜ ~ N(0, 0.01), drop first 100 as burn-in  
2) Compute inflation πₜ and unemployment uₜ under constant, adaptive, and rational expectations  
3) Produce two figures, save them as PNGs, and print their file paths:
     • simulation_panels.png  
     • correlation_chart.png  
Both figures include a centered, black citation at the bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def simulate_series(N=1000, burn=100, alpha=1.5, u_n=0.05, mu=0.0):
    # Draw shocks and simulate inflation πₜ = πₜ₋₁ + μ + εₜ
    eps = np.random.randn(N) * np.sqrt(0.01)
    pi  = np.zeros(N+1)
    for t in range(1, N+1):
        pi[t] = pi[t-1] + mu + eps[t-1]
    # Drop π₋₁ and burn-in
    pi  = pi[1:][burn:]
    eps = eps[burn:]
    # Build expectations
    pi_e_const    = np.zeros_like(pi)
    pi_e_adaptive = np.concatenate(([0.0], pi[:-1]))
    pi_e_rational = pi_e_adaptive + mu
    # Compute unemployment paths
    u_const    = u_n + (pi_e_const    - pi) / alpha
    u_adaptive = u_n + (pi_e_adaptive - pi) / alpha
    u_rational = u_n + (pi_e_rational - pi) / alpha
    return pi, u_const, u_adaptive, u_rational

def main():
    # Styling and reproducibility
    sns.set_style("whitegrid")
    np.random.seed(42)

    # Run simulation
    pi, u_const, u_adaptive, u_rational = simulate_series()
    T = len(pi)
    t = np.arange(1, T+1)
    citation = "Calculations by Theodoros Kourtalis for MT2 – 1412"

    # ─── Figure 1: Inflation & Unemployment Panels ─────────────────────────
    fig1, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    axs[0].plot(t, pi,        color='black', lw=1)
    axs[0].set_title("Simulated Inflation")
    axs[0].set_ylabel(r"$\pi_t$")

    axs[1].plot(t, u_const,   color='C0', lw=1)
    axs[1].axhline(0.05, color='gray', ls='--')
    axs[1].set_title("Unemployment: Constant Expectations")
    axs[1].set_ylabel(r"$u_t$")

    axs[2].plot(t, u_adaptive, color='gray', lw=1)
    axs[2].axhline(0.05, color='black', ls='--')
    axs[2].set_title("Unemployment: Adaptive Expectations")
    axs[2].set_ylabel(r"$u_t$")

    axs[3].plot(t, u_rational, color='green', lw=1)
    axs[3].axhline(0.05, color='gray', ls='--')
    axs[3].set_title("Unemployment: Rational Expectations")
    axs[3].set_xlabel("Period $t$")
    axs[3].set_ylabel(r"$u_t$")

    # Centered citation at bottom
    fig1.text(0.5, 0.01, citation, ha='center', va='bottom',
              fontsize=9, color='black')
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save and report
    fname1 = "simulation_panels.png"
    fig1.savefig(fname1, dpi=300, bbox_inches='tight')
    print(f"Saved simulation panels to: {os.path.abspath(fname1)}")

    # ─── Figure 2: Correlation Bar Chart ───────────────────────────────────
    corrs = [
        np.corrcoef(pi, u_const)[0,1],
        np.corrcoef(pi, u_adaptive)[0,1],
        np.corrcoef(pi, u_rational)[0,1],
    ]
    labels = ["Constant", "Adaptive", "Rational"]
    colors = ["C0", "gray", "green"]
    y_pos  = np.arange(len(labels))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.barh(y_pos, corrs, color=colors, edgecolor='k', height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.set_xlabel(r"Pearson Correlation $\rho(\pi_t, u_t)$", fontsize=12)
    ax2.set_title("Correlation between Inflation and Unemployment", fontsize=14)

    # Pad x-axis so annotations fit
    mc, Mc = min(corrs), max(corrs)
    ax2.set_xlim(mc - 0.1, Mc + 0.1)

    # Annotate each bar
    for bar, c in zip(bars, corrs):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        ha = 'right' if x < 0 else 'left'
        dx = -0.02 if x < 0 else 0.02
        ax2.text(x + dx, y, f"{c:.2f}", va='center', ha=ha, fontsize=11)

    ax2.axvline(0, color='black', lw=1)
    sns.despine(left=True, bottom=False)

    # Centered citation
    fig2.text(0.5, 0.01, citation, ha='center', va='bottom',
              fontsize=9, color='black')
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save and report
    fname2 = "correlation_chart.png"
    fig2.savefig(fname2, dpi=300, bbox_inches='tight')
    print(f"Saved correlation chart to: {os.path.abspath(fname2)}")

    plt.show()

if __name__ == "__main__":
    main()