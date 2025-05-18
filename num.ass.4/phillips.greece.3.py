#!/usr/bin/env python3
"""
Enhanced Phillips Curve Plotter for Greece with dynamic recessions, combined non-crisis and subperiod panels,
per-period refitting, parameter listing in each plot via legends, and detailed annotations.
"""

import datetime
import logging
import math
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================= User Settings =================
START_YEAR = 1999
END_YEAR   = 2024  # Inclusive; adjust as needed
THRESHOLD  = 5     # Minimum data points per panel

# Original Phillips curve parameters
ORIG_PARAMS = (0.900, 9.638, -1.394)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# ================= Data Fetching =================
def fetch_data(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    logging.info(f"Fetching Greece data from {start.date()} to {end.date()}")
    series = ['LRHUTTTTGRM156S', 'GRCCPIALLMINMEI']
    df = pdr.DataReader(series, 'fred', start, end)
    df.columns = ['Unemployment', 'CPI']
    df['Inflation'] = df['CPI'].pct_change(12) * 100
    return df.dropna()

# ================= Recession Detection =================
def fetch_recessions(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    logging.info(f"Fetching Greece recession indicator from {start.date()} to {end.date()}")
    rec = pdr.DataReader('GRCRECM', 'fred', start, end).dropna()
    rec['prev'] = rec['GRCRECM'].shift(1).fillna(0).astype(int)
    starts = rec[(rec['GRCRECM']==1)&(rec['prev']==0)].index
    ends   = rec[(rec['GRCRECM']==1)&(rec['GRCRECM'].shift(-1).fillna(0)==0)].index
    periods = []
    for idx, s in enumerate(starts):
        e = ends[idx] if idx < len(ends) else rec.index[-1]
        periods.append({'Name': f"Recession {idx+1}: {s.strftime('%b %Y')}–{e.strftime('%b %Y')}", 'start': s, 'end': e})
    return pd.DataFrame(periods)

# ================= Model Definition =================
def phillips_model(u, a, b, c):
    return b * u**c - a

# ================= Curve Fitting =================
def fit_curve(x, y, p0, bounds):
    if len(x) <= THRESHOLD:
        return None
    params, _ = curve_fit(phillips_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
    logging.info(f"Fitted params: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}")
    return params

# ================= Plotting =================
def plot_results(df, recessions, u_range, start, end, p0, bounds):
    """Generate panels: all data, combined non-crisis, and each recession period, with legends showing parameter values."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Build mask for recessions
    recession_masks = []
    for _, r in recessions.iterrows():
        mask = (df.index >= r['start']) & (df.index <= r['end'])
        recession_masks.append((r['Name'], mask))

    # Combined non-crisis mask is complement of any recession mask
    combined_mask = pd.Series(True, index=df.index)
    for _, mask in recession_masks:
        combined_mask &= ~mask

    # Define panels
    panels = [{'Name': 'All data', 'mask': pd.Series(True, index=df.index)},
              {'Name': 'All non-recessions', 'mask': combined_mask}]
    panels += [{'Name': name, 'mask': pd.Series(mask, index=df.index)} for name, mask in recession_masks]

    # Filter panels with sufficient data
    valid = [p for p in panels if p['mask'].sum() > THRESHOLD]
    n = len(valid)
    cols, rows = 2, math.ceil(n / 2)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab10')

    # Plot each valid panel
    for i, p in enumerate(valid):
        ax = axes[i]
        sub = df[p['mask']]
        color = cmap(i % 10)
        ax.scatter(sub['Unemployment'], sub['Inflation'], color=color, alpha=0.7, edgecolor='k', label='Data')

        # Plot original Phillips curve with legend
        orig_label = f"Original: a={ORIG_PARAMS[0]:.3f}, b={ORIG_PARAMS[1]:.3f}, c={ORIG_PARAMS[2]:.3f}"
        ax.plot(u_range, phillips_model(u_range, *ORIG_PARAMS), '--', color='gray', label=orig_label)

        # Per-panel refit and legend
        pr = fit_curve(sub['Unemployment'].values, sub['Inflation'].values, p0, bounds)
        if pr is not None:
            fit_label = f"Fitted: a={pr[0]:.3f}, b={pr[1]:.3f}, c={pr[2]:.3f}"
            ax.plot(u_range, phillips_model(u_range, *pr), '-', color='black', lw=2, label=fit_label)

        ax.set_title(p['Name'])
        ax.set_xlabel('Unemployment (%)')
        ax.set_ylabel('Inflation (%)')
        ax.set_xlim(sub['Unemployment'].min() * 0.95, sub['Unemployment'].max() * 1.05)
        ax.set_ylim(sub['Inflation'].min() * 0.95, sub['Inflation'].max() * 1.05)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    # Remove extra axes
    for j in range(n, rows * cols):
        fig.delaxes(axes[j])

    # Print fitted parameters
    print("Fitted parameters by period:")
    for name, pr in (
        (p['Name'], fit_curve(df.loc[p['mask'], 'Unemployment'].values,
                               df.loc[p['mask'], 'Inflation'].values,
                               p0, bounds))
        for p in valid
    ):
        if pr is not None:
            a, b, c = pr
            print(f"  {name}: a={a:.3f}, b={b:.3f}, c={c:.3f}")

    # Annotation
    earliest = df.index.min().strftime('%Y-%m')
    latest = df.index.max().strftime('%Y-%m')
    freq = pd.infer_freq(df.index) or 'Monthly'
    annotation = (
        f"Dataset covers: {earliest} to {latest} | Frequency: {freq} | "
        "Unemployment/CPI source: FRED (LRHUTTTTGRM156S, GRCCPIALLMINMEI) | "
        "Recession source: FRED (GRCRECM) | Calculations by Theodoros Kourtalis"
    )
    fig.suptitle(f'Greece Phillips Curve Subperiods ({start.year}–{end.year})', fontsize=16)
    fig.text(0.5, 0.02, annotation, ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure to file
    filename = f'greece_phillips_curve_{start.year}_{end.year}.png'
    fig.savefig(filename, dpi=300)
    logging.info(f"Saved figure to {filename}")

# ================= Main
if __name__ == '__main__':
    s = datetime.datetime(START_YEAR, 1, 1)
    e = datetime.datetime(END_YEAR, 12, 31)
    df = fetch_data(s, e)
    recs = fetch_recessions(s, e)
    p0 = [0.5, 5.0, -1.0]
    bounds = ([-5, 0, -5], [5, 50, 5])
    u_range = np.linspace(df['Unemployment'].min(), df['Unemployment'].max(), 200)
    plot_results(df, recs, u_range, s, e, p0, bounds)
