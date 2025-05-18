#!/usr/bin/env python3
"""
Enhanced U.S. Phillips Curve Plotter with dynamic recessions, combined non-recession panel,
subperiod panels, per-period refitting, legends showing original and fitted parameters,
and detailed annotations including credit.
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
END_YEAR = 2025  # Inclusive
THRESHOLD = 5    # Minimum data points per period

# Original Phillips curve parameters
ORIG_PARAMS = (0.900, 9.638, -1.394)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# ================= Data Fetching =================
def fetch_data(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """Fetch UNRATE and CPIAUCSL from FRED and compute YoY inflation."""
    logging.info("Fetching FRED data from %s to %s", start.date(), end.date())
    df = pdr.DataReader(['UNRATE', 'CPIAUCSL'], 'fred', start, end)
    df.columns = ['Unemployment', 'CPI']
    df['Inflation'] = df['CPI'].pct_change(12) * 100
    return df.dropna(subset=['Inflation'])

# ================= Recession Detection =================
def fetch_crises(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """Fetch USREC recession indicator and identify recession spells overlapping the window."""
    logging.info("Fetching USREC data from %s to %s", start.date(), end.date())
    rec = pdr.DataReader('USREC', 'fred', start, end).dropna()
    rec['prev'] = rec['USREC'].shift(1).fillna(0).astype(int)
    starts = rec[(rec['USREC'] == 1) & (rec['prev'] == 0)].index
    ends   = rec[(rec['USREC'] == 1) & (rec['USREC'].shift(-1).fillna(0).astype(int) == 0)].index
    records = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else rec.index[-1]
        records.append({'Name': f"Recession {i+1}: {s.strftime('%b %Y')}–{e.strftime('%b %Y')}",
                        'start': s, 'end': e})
    # Non-recession runs
    non_mask = pd.Series(True, index=rec.index)
    for r in records:
        non_mask &= ~((rec.index >= r['start']) & (rec.index <= r['end']))
    non_runs = []
    curr = None
    for date, flag in non_mask.items():
        if flag and curr is None:
            curr = date
        if not flag and curr is not None:
            non_runs.append((curr, date - pd.offsets.MonthBegin(1)))
            curr = None
    if curr is not None:
        non_runs.append((curr, rec.index[-1]))
    for j, (ns, ne) in enumerate(non_runs):
        records.append({'Name': f"Non-recession {j+1}: {ns.strftime('%b %Y')}–{ne.strftime('%b %Y')}",
                        'start': ns, 'end': ne})
    return pd.DataFrame(records)

# ================= Model Definition =================
def phillips_model(u, a, b, c):
    """Power-law Phillips curve: inflation = b * u^c - a."""
    return b * u**c - a

# ================= Curve Fitting =================
def fit_curve(x: np.ndarray, y: np.ndarray, p0, bounds, maxfev=10000):
    """Fit the power-law model if enough data points."""
    if len(x) <= THRESHOLD:
        return None
    params, _ = curve_fit(phillips_model, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    logging.info("Fitted params: a=%.3f, b=%.3f, c=%.3f", *params)
    return params

# ================= Plotting =================
def plot_results(df: pd.DataFrame, crises: pd.DataFrame,
                 u_range: np.ndarray, start: datetime.datetime, end: datetime.datetime,
                 p0, bounds):
    """Plot panels for all data, combined non-recession, and subperiods with legends."""
    plt.style.use('seaborn-v0_8-whitegrid')
    # Label periods
    df['Crisis'] = 'Non-crisis'
    for _, rec in crises.iterrows():
        mask = (df.index >= rec['start']) & (df.index <= rec['end'])
        df.loc[mask, 'Crisis'] = rec['Name']

    # Build list: All data, All non-recession, then each subperiod
    periods = [{'Name': 'All data', 'mask': df.index >= df.index.min()}]
    mask_non_all = df['Crisis'].str.startswith('Non-recession')
    periods.append({'Name': 'All non-recession', 'mask': mask_non_all})
    for _, rec in crises.iterrows():
        periods.append({'Name': rec['Name'],
                         'mask': (df.index >= rec['start']) & (df.index <= rec['end'])})

    # Filter valid periods
    valid = [p for p in periods if p['mask'].sum() > THRESHOLD]
    n = len(valid)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    cmap = plt.get_cmap('tab10')

    # Plot each valid panel
    for idx, p in enumerate(valid):
        ax = axes_flat[idx]
        sub = df.loc[p['mask']]
        color = cmap(idx % 10)
        ax.scatter(sub['Unemployment'], sub['Inflation'], color=color, alpha=0.7, edgecolor='k', label='Data')

        # Original Phillips curve
        orig_label = f"Original: a={ORIG_PARAMS[0]:.3f}, b={ORIG_PARAMS[1]:.3f}, c={ORIG_PARAMS[2]:.3f}"
        ax.plot(u_range, phillips_model(u_range, *ORIG_PARAMS), '--', color='gray', label=orig_label)

        # Per-panel refit
        params = fit_curve(sub['Unemployment'].values, sub['Inflation'].values, p0, bounds)
        if params is not None:
            fit_label = f"Fitted: a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}"
            ax.plot(u_range, phillips_model(u_range, *params), '-', color='black', lw=2, label=fit_label)

        ax.set_title(p['Name'])
        ax.set_xlabel('Unemployment (%)')
        ax.set_ylabel('Inflation (YoY %)')
        ax.set_xlim(sub['Unemployment'].min() * 0.95, sub['Unemployment'].max() * 1.05)
        ax.set_ylim(sub['Inflation'].min() * 0.95, sub['Inflation'].max() * 1.05)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    # Remove extra axes
    for j in range(n, rows * cols):
        fig.delaxes(axes_flat[j])

    # Display fitted parameters
    print("Fitted parameters by period:")
    for name, pr in ({p['Name']: pr for p, pr in zip(valid, [fit_curve(df.loc[p['mask'], 'Unemployment'].values,
                                                                        df.loc[p['mask'], 'Inflation'].values,
                                                                        p0, bounds)
                                                                 for p in valid])}).items():
        if pr is not None:
            a, b, c = pr
            print(f"  {name}: a={a:.3f}, b={b:.3f}, c={c:.3f}")

    # Annotation
    earliest = df.index.min().strftime('%Y-%m')
    latest = df.index.max().strftime('%Y-%m')
    freq = pd.infer_freq(df.index) or 'Monthly'
    annotation = (
        f"Dataset covers: {earliest} to {latest} | Frequency: {freq} | "
        "Data source: FRED (UNRATE, CPIAUCSL) | Recession series: USREC | Calculations by Theodoros Kourtalis"
    )
    fig.suptitle(f'U.S. Phillips Curve Subperiods ({start.year}–{end.year})', fontsize=16)
    fig.text(0.5, 0.02, annotation, ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure to file
    filename = f'us_phillips_curve_{start.year}_{end.year}.png'
    fig.savefig(filename, dpi=300)
    logging.info(f"Saved figure to {filename}")

    # Show plot
    plt.show()

# ================= Main Execution =================
if __name__ == '__main__':
    s = datetime.datetime(START_YEAR, 1, 1)
    e = datetime.datetime(END_YEAR, 12, 31)
    df = fetch_data(s, e)
    crises = fetch_crises(s, e)
    p0 = [0.9, 9.638, -1.394]
    bounds = ([-10, 0, -5], [10, 50, 5])
    u_range = np.linspace(df['Unemployment'].min(), df['Unemployment'].max(), 300)
    plot_results(df, crises, u_range, s, e, p0, bounds)
