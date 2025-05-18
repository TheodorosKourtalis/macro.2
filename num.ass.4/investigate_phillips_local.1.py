#!/usr/bin/env python3
"""
investigate_phillips_local.py

Analyzes how the Phillips Curve slope depends on expectation formation (Adaptive vs. Rational proxies),
using the formula:
    π_t - π^e_t = -α (u_t - u_n)

This script computes the ‘‘inflation surprise’’ for each proxy, measures the unemployment gap,
estimates α for different periods, and produces a clear bar plot comparing slopes (full sample),
annotating each country with its actual sample years.
"""
import argparse
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
import statsmodels.formula.api as smf
from pandas_datareader import data as pdr

# ----------------- Default paths -----------------
DEFAULT_EXPECT = {
    'US': '/Users/thodoreskourtales/macro.num_ass_4/US/expectations_US.csv',
    'Greece': '/Users/thodoreskourtales/macro.num_ass_4/GR/expectations_GR.csv'
}

# ----------------- Command-line args -----------------
parser = argparse.ArgumentParser(
    description="Estimate Phillips Curve slope under different expectation proxies"
)
parser.add_argument(
    "--expect-us", dest='expect_us',
    default=DEFAULT_EXPECT['US'],
    help="Path to US expectations CSV (default from config)"
)
parser.add_argument(
    "--expect-gr", dest='expect_gr',
    default=DEFAULT_EXPECT['Greece'],
    help="Path to Greece expectations CSV (default from config)"
)
args = parser.parse_args()

# Validate files (warn if missing)
for loc, path in [('US', args.expect_us), ('Greece', args.expect_gr)]:
    if not os.path.isfile(path):
        print(f"Warning: {loc} expectation file not found at {path}. Check the path.")

# ----------------- Configuration -----------------
COUNTRY_FRED = {
    'US': {
        'cpi': 'CPIAUCSL',
        'unemp': 'UNRATE',
        'expect_csv': args.expect_us
    },
    'Greece': {
        'cpi': 'CP0000GRM086NEST',
        'unemp': 'LRHUTTTTGRM156N',
        'expect_csv': args.expect_gr
    }
}
START = datetime.datetime(1999, 1, 1)
END   = datetime.datetime(2025, 12, 31)
BREAK_YEAR = 2009
MIN_OBS = 15
OUT_CSV = 'phillips_alpha_summary.csv'
OUT_PLOT = 'phillips_alpha_comparison.png'

# ---------- Helper functions -----------

def fetch_fred_series(codes):
    df = pdr.DataReader([codes['cpi'], codes['unemp']], 'fred', START, END)
    df.columns = ['CPI', 'Unemployment']
    return df.resample('MS').last()

def load_country(name, config):
    # Fetch and compute inflation & unemployment gap
    df = fetch_fred_series(config)
    df['Inflation'] = df['CPI'].pct_change(12) * 100
    u_n = df['Unemployment'].mean()
    df['U_gap'] = df['Unemployment'] - u_n

    # Load expectations
    try:
        df_exp = pd.read_csv(config['expect_csv'], parse_dates=['ds']).set_index('ds')
    except FileNotFoundError:
        print(f"Error loading expectations for {name} from {config['expect_csv']}")
        sys.exit(1)
    for col in ['y', 'exp_mean', 'adaptive']:
        if col not in df_exp.columns:
            raise KeyError(f"Required column '{col}' missing in {config['expect_csv']}")

    # Build proxies dictionary
    proxies = {'exp_rational': 'exp_mean', 'exp_adaptive': 'adaptive'}
    if name == 'US' and 'mich' in df_exp.columns:
        proxies['exp_mich'] = 'mich'

    # Merge and rename
    df = df.join(df_exp[list(proxies.values())], how='inner')
    df.rename(columns={v: k for k, v in proxies.items()}, inplace=True)

    # Compute surprises
    for key in proxies:
        code = key.split('_')[1]
        df[f"surprise_{code}"] = df['Inflation'] - df[key]

    # Drop any rows missing required series
    required = ['Inflation', 'U_gap'] + [f"surprise_{key.split('_')[1]}" for key in proxies]
    df = df.dropna(subset=required)

    df['Year'] = df.index.year
    df['Country'] = name
    return df

def estimate_alpha(df, var):
    model = smf.ols(f"{var} ~ U_gap", data=df).fit(cov_type='HC1')
    alpha = -model.params['U_gap']
    return alpha, model.bse['U_gap'], model.pvalues['U_gap'], model.rsquared

# ----------------- Main computation -----------------
results = []
for country, cfg in COUNTRY_FRED.items():
    df_country = load_country(country, cfg)
    periods = {
        'Full': (START.year, END.year),
        'Pre-GFC': (START.year, BREAK_YEAR - 1),
        'Post-GFC': (BREAK_YEAR, END.year)
    }
    available = [c for c in df_country.columns if c.startswith('surprise_')]
    labels = {
        'surprise_rational': 'Rational',
        'surprise_adaptive': 'Adaptive',
        'surprise_mich': 'Michigan'
    }

    for pname, (y0, y1) in periods.items():
        sub = df_country[(df_country['Year'] >= y0) & (df_country['Year'] <= y1)]
        if len(sub) < MIN_OBS:
            continue
        for var in available:
            alpha, se, pval, r2 = estimate_alpha(sub, var)
            results.append({
                'Country': country,
                'Period': pname,
                'Expectation': labels.get(var, var),
                'Alpha': alpha,
                'SE': se,
                'p_value': pval,
                'R2': r2
            })

# Save summary CSV
df_res = pd.DataFrame(results)
df_res.to_csv(OUT_CSV, index=False)
print(f"Summary saved to {OUT_CSV}")

# ——— Plot only full-sample slopes, with sample-period labels ———
full = df_res[df_res['Period'] == 'Full']
pivot = full.pivot(index='Country', columns='Expectation', values='Alpha')

# Compute actual sample period for each country
sample_periods = {}
for country, cfg in COUNTRY_FRED.items():
    df_ct = load_country(country, cfg)
    years = df_ct.index.year
    sample_periods[country] = (years.min(), years.max())

# Build custom x-tick labels like "US (1999–2025)", "Greece (2005–2021)"
xtick_labels = [
    f"{country} ({sample_periods[country][0]}–{sample_periods[country][1]})"
    for country in pivot.index
]

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
pivot.plot(kind='bar', rot=0, ax=ax)
ax.set_xticklabels(xtick_labels)
ax.set_ylabel(r'Estimated $\alpha$')
ax.set_title(
    r'Phillips Curve Slope $\alpha$ under Different Expectations'
)

# Add citation text at bottom
citation = (
    "Sources: Federal Reserve Bank of St. Louis (FRED); "
    "Author’s calculations by Theodoros Kourtalis. "
    "Adaptive Expectations represent a one-period lag, "
    "Rational Expectations derived via a Bayesian Prophet ensemble."
)
fig.text(0.5, -0.05, citation, ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PLOT, bbox_inches='tight')
print(f"Plot saved to {OUT_PLOT}")