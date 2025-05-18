#!/usr/bin/env python3
"""
expected_inflation_ra.py — v1.2
Proxy for Rational-Expectations (RE) inflation using a Prophet ensemble.
Adds benchmarks and plots with Michigan and Adaptive expectations.
"""

from __future__ import annotations
import argparse
import os
import datetime as dt
import warnings
import joblib
import numpy as np
import pandas as pd
from pandas_datareader import data as web
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, DateOffset
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib.lines import Line2D

CB_PAL = sns.color_palette("tab10", 4)
# Define consistent palette colors
COL_ACTUAL = CB_PAL[0]
COL_RE = CB_PAL[1]
COL_ADAPTIVE = CB_PAL[2]
COL_MICHIGAN = CB_PAL[3]

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 120
# plt.style.use('ggplot')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

#
# Configuration
# Directory for all outputs to avoid overwriting previous runs
OUTPUT_DIR = os.path.dirname(__file__)
#
# ------------------------------------------------------------------
# Country-specific series configurations
COUNTRY_CONFIG = {
    "US": {
        "series": {
            "cpi":       "CPIAUCSL",
            "core_cpi":  "CPILFESL",
            "unemp_rate":"UNRATE",
            "fed_funds": "FEDFUNDS",
            "oil_price": "DCOILWTICO",
            "m2":        "M2SL",
            "ppi":       "PPIACO",
            "sentiment": "UMCSENT",
            "mich":      "MICH",     # University of Michigan: Inflation Expectation [MICH]
        },
        # No survey key
    },
    "GR": {
        "series": {
            "cpi":        "CP0000GRM086NEST",   # HICP: All Items, Greece
            "core_cpi":   "CPGRLE01GRM657N",    # HICP ex-Food & Energy, Greece  [oai_citation:3‡FRED](https://fred.stlouisfed.org/series/CPGRLE01GRM657N?utm_source=chatgpt.com)
            "unemp_rate": "LRHUTTTTGRM156N",    # OECD monthly unemployment, Greece  [oai_citation:4‡FRED](https://fred.stlouisfed.org/series/CPGRLE01GRM657N?utm_source=chatgpt.com)
            "fed_funds":  "ECBMRRFR",           # ECB main refinancing rate
            "oil_price":  "DCOILBRENTEU",         # WTI oil price
            "m2":         "WM2NS",              # Euro-area broad money (proxy for GR M2)  [oai_citation:5‡FRED](https://fred.stlouisfed.org/series/WM2NS?utm_source=chatgpt.com)
            "ppi":        "PIEATI02GRM661N",    # OECD Industrial PPI, Greece
            "sentiment":  "CSCICP02GRM460S",    # OECD consumer confidence, Greece
        },
        "survey": None
    }
}
# ------------------------------------------------------------------
CONFINT = 0.8
CV_YEARS = 10
MAX_MODELS = 4
BO_INIT_POINTS = 5
BO_N_ITER = 20
ROLL_ERR_WIN = 36
# Forecast and adaptive horizon in months
HORIZON_MONTHS = 12

MICHIGAN_MAIZE = "#FFCB05"
# ------------------------------------------------------------------
# Signature colors used consistently across all figures
COL_TRUE        = "black"       # Actual inflation
COL_RE_OOS      = "green"       # Rational Expectations – out‑of‑sample
COL_RE_INSAMPLE = "blue"        # Rational Expectations – in‑sample
COL_ADAPTIVE    = "gray"        # Adaptive expectations
COL_MICHIGAN    = MICHIGAN_MAIZE  # Michigan survey
# ------------------------------------------------------------------

# Data helpers

def fetch_series(code: str, start: str, end: str, vintage: bool) -> pd.Series:
    src = "alfred" if vintage else "fred"
    df = web.DataReader(code, src, start, end)
    ser = df.droplevel(0, axis=1).iloc[:, 0] if vintage else df.iloc[:, 0]
    return ser.resample("MS").last()


def fetch_all(series_map: dict[str,str], start: str, end: str, vintage: bool=False) -> pd.DataFrame:
    data = {}
    for name, code in series_map.items():
        try:
            data[name] = fetch_series(code, start, end, vintage)
        except Exception as e:
            print(f"⚠️ failed to fetch {name}: {e}")
    if "cpi" not in data:
        raise ValueError("CPI missing after fetch.")
    idx = data["cpi"].index
    df = pd.DataFrame({n: s.reindex(idx) for n, s in data.items()})
    df["y"] = df["cpi"].pct_change(12) * 100
    # forward/backfill all other series
    for n in data:
        if n != "cpi":
            df[n] = df[n].ffill().bfill()
    df = df.dropna(subset=["y"]).reset_index()
    df.rename(columns={df.columns[0]: "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
    return df[["ds", "y"] + [n for n in data if n != "cpi"]]

# Prophet factory

def make_prophet(cps: float, seas: float) -> Prophet:
    m = Prophet(growth="linear", interval_width=CONFINT, yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=cps, seasonality_prior_scale=seas)
    m.add_country_holidays(country_name="US")
    m.add_seasonality("quarterly", period=91.3, fourier_order=5)
    # REGRESSORS is now country-dependent; will be set in process_country
    for r in getattr(make_prophet, "REGRESSORS", []):
        m.add_regressor(r, prior_scale=5)
    return m

# Hyper-parameter search

def bayes_search(train: pd.DataFrame) -> list[tuple[float, float]]:
    def cv_score(cps: float, seas: float) -> float:
        m = make_prophet(cps, seas)
        m.fit(train)
        try:
            cv = cross_validation(m, horizon="365 days",
                                   initial=pd.Timedelta(days=max(365*5, int(len(train)*0.6)*30)),
                                   period="365 days", parallel=None, disable_tqdm=True)
            mape = performance_metrics(cv)["mape"].iloc[-1]
        except Exception:
            mape = (train["y"] - m.predict(train)["yhat"]).abs().mean()
        return -mape
    optimizer = BayesianOptimization(f=cv_score, pbounds={"cps":(0.02,1.0),"seas":(1.0,15.0)}, random_state=0, verbose=0)
    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)
    top = sorted(optimizer.res, key=lambda r: -r["target"])[:MAX_MODELS]
    return [(r["params"]["cps"], r["params"]["seas"]) for r in top]

def get_nber_recessions(start, end):
    rec = web.DataReader('USREC', 'fred', start, end)['USREC']
    rec = rec.resample('MS').last()
    return [date for date, in_rec in rec.items() if in_rec == 1]

# Random-walk forecast

def fc_regressors_random_walk(hist: pd.DataFrame) -> pd.DataFrame:
    # now forecast exactly HORIZON_MONTHS months ahead
    forecast_horizon = hist["ds"].max() + DateOffset(months=HORIZON_MONTHS)
    vals = hist.iloc[-1][getattr(fc_regressors_random_walk, "REGRESSORS", [])]
    return pd.DataFrame({"ds": [forecast_horizon], **vals.to_dict()})

# Metrics

def compute_metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = (true - pred).abs()
    return {"MAPE": (err/true.abs()).mean()*100, "RMSE": np.sqrt(((true-pred)**2).mean())}

# Walk-forward

def walk_forward(df: pd.DataFrame, params_pool: list[tuple[float,float]], force: bool = False) -> pd.DataFrame:
    recs = []
    first_cut = df["ds"].min() + pd.DateOffset(years=CV_YEARS)
    cuts = df[df["ds"]>=first_cut]["ds"].iloc[:-1]
    for cut in cuts:
        hist = df[df["ds"]<=cut]
        future = fc_regressors_random_walk(hist)
        preds, weights = [], []
        for cps, seas in params_pool:
            key = f"{cut.strftime('%Y-%m')}_{cps:.3f}_{seas:.3f}.joblib"
            path = os.path.join(CACHE_DIR, key)
            if os.path.exists(path) and not force:
                m = joblib.load(path)
            else:
                m = make_prophet(cps, seas)
                m.fit(hist)
                joblib.dump(m, path)
            yhat = m.predict(future)["yhat"].iloc[0]
            err = (hist["y"] - m.predict(hist)["yhat"]).abs()
            preds.append(yhat); weights.append(1/err.tail(ROLL_ERR_WIN).mean())
        # record forecast back at the "as-of" cut date
        recs.append({
            "ds": cut,
            "exp_mean": np.average(preds, weights=weights),
            **{f"model_{i+1}": p for i, p in enumerate(preds)}
        })
    return pd.DataFrame(recs).set_index("ds")

# CLI / main

def parse_args():
    today = dt.date.today().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser("Expected inflation RE proxy")
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default=today)
    p.add_argument("--vintage", choices=["on","off"], default="off")
    p.add_argument("--plot", choices=["on","off"], default="on")
    p.add_argument("--force", action="store_true",
                   help="Re-run modeling and plotting even if cached outputs exist")
    return p.parse_args()

def _generate_plots_and_stats(country_code: str, cfg: dict, args, df: pd.DataFrame, merged: pd.DataFrame, pool: list[tuple[float, float]]):
    """
    Generate all required plots and error statistics from merged expectations,
    and adaptive series.
    """
    country_out = os.path.join(OUTPUT_DIR, country_code)
    # Citation text
    citation = (
        "Sources: Federal Reserve Bank of St. Louis (FRED); "
        "Author’s calculations by Theodoros Kourtalis. "
        "Adaptive Expectations represent a one-period lag, Rational Expectations derived via a Bayesian Prophet ensemble."
    )

    # Determine if full data available
    full = (df is not None and pool is not None)

    # 1) Actual vs All Proxies
    fig, ax = plt.subplots(figsize=(12, 6))
    # Determine key dates
    first_oos = merged['exp_mean'].first_valid_index()           # first date an OOS forecast appears
    earliest  = df['ds'].min() if df is not None else merged.index.min()
    max_ds    = df['ds'].max() if df is not None else merged.index.max()

    # Mark and shade the out‑of‑sample period
    ax.axvline(first_oos, color="black", linestyle=":", linewidth=1)
    ax.axvspan(first_oos, max_ds, color="lightgrey", alpha=0.3)

    # Recession shading
    rec_start_list = get_nber_recessions(earliest, max_ds)
    for rec_start in rec_start_list:
        ax.axvspan(rec_start, rec_start + MonthEnd(1), color="gray", alpha=0.2)

    # Ensure the x‑axis covers the whole data range
    ax.set_xlim(earliest, max_ds)

    # Plot actual π from df if available, otherwise from merged
    if df is not None:
        actual_series = df.set_index('ds')['y']
    else:
        actual_series = merged['y']
    ax.plot(actual_series,          color=COL_TRUE,     linewidth=2, linestyle='-', label='Actual π')
    ax.plot(merged['exp_mean'],     color=COL_RE_OOS,   linewidth=2, linestyle=':', label='R.E π')
    ax.plot(merged['adaptive'],     color=COL_ADAPTIVE, linewidth=2, linestyle=':', label='Adaptive π')
    if 'mich' in merged.columns:
        ax.plot(merged['mich'],     color=COL_MICHIGAN, linewidth=2, linestyle=':', label='Michigan π')
    ax.set(title=f"Inflation Expectations vs Actual ({country_code})",
           xlabel="Date", ylabel="YoY inflation (%)")
    ax.legend(loc='upper left', frameon=True)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig.text(0.5, 0.01, citation, ha='center', va='top', fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(os.path.join(country_out, "actual_vs_all_proxies.png"),
                bbox_inches='tight', pad_inches=0.1)
    print("▸ Saved plot: actual_vs_all_proxies.png")
    plt.show()
    plt.close(fig)

    # Compute in-sample ensemble predictions and 2) In-sample vs Actual only if full data is available
    if full:
        # Compute in-sample ensemble predictions on full sample
        in_preds = []
        weights = []
        for cps, seas in pool:
            m = make_prophet(cps, seas)
            m.fit(df)
            yhat = m.predict(df)[['ds','yhat']].set_index('ds')['yhat']
            in_preds.append(yhat)
            err = (df.set_index('ds')['y'] - yhat).abs()
            weights.append(1/err.tail(ROLL_ERR_WIN).mean())
        preds_df = pd.concat(in_preds, axis=1)
        w = np.array(weights)
        if w.sum() > 0:
            df['in_sample'] = np.average(preds_df.values, axis=1, weights=w)
        else:
            df['in_sample'] = preds_df.mean(axis=1)

        # 2) In-sample vs Actual
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.set_index('ds')['y'],         color=COL_TRUE,        linewidth=2, linestyle='-', label='Actual π')
        ax2.plot(df.set_index('ds')['in_sample'], color=COL_RE_INSAMPLE, linewidth=2, linestyle=':', label='R.E π (In‑Sample)')
        ax2.set(title=f"In-sample Fit vs Actual ({country_code})",
               xlabel="Date", ylabel="YoY inflation (%)")
        ax2.legend(loc='upper left', frameon=True)
        ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        fig2.text(0.5, 0.01, citation, ha='center', va='top', fontsize=10, wrap=True)
        fig2.tight_layout()
        fig2.savefig(os.path.join(country_out, "actual_vs_Rational Exp (IN).png"),
                    bbox_inches='tight', pad_inches=0.1)
        print("▸ Saved plot: actual_vs_Rational Exp (IN).png")
        plt.show()
        plt.close(fig2)

    # 3) Adaptive vs Rational Exp (OOS)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(merged['y'],        color=COL_TRUE,     linewidth=2, linestyle='-', label='Actual π')
    ax4.plot(merged['adaptive'], color=COL_ADAPTIVE, linewidth=2, linestyle=':', label='Adaptive π')
    ax4.plot(merged['exp_mean'], color=COL_RE_OOS,   linewidth=2, linestyle=':', label='R.E π')
    ax4.set(title=f"Adaptive vs Rational Exp (OOS) ({country_code})",
            xlabel="Date", ylabel="YoY inflation (%)")
    ax4.legend(loc='upper left', frameon=True)
    ax4.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig4.text(0.5, 0.01, citation, ha='center', va='top', fontsize=10, wrap=True)
    fig4.tight_layout()
    fig4.savefig(os.path.join(country_out, "adaptive_vs_Rational_Exp(OOS).png"),
                bbox_inches='tight', pad_inches=0.1)
    print("▸ Saved plot: adaptive_vs_Rational_Exp(OOS).png")
    plt.show()
    plt.close(fig4)

    # Plot Michigan vs Forecast only if present
    if 'mich' in merged.columns:
        # 4) Michigan vs Forecast
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(merged['y'],        color=COL_TRUE,     linewidth=2, linestyle='-', label='Actual π')
        ax5.plot(merged['exp_mean'], color=COL_RE_OOS,   linewidth=2, linestyle=':', label='R.E π')
        ax5.plot(merged['mich'],     color=COL_MICHIGAN, linewidth=2, linestyle=':', label='Michigan π')
        ax5.set(title=f"Michigan vs Forecast (OOS) ({country_code})",
                xlabel="Date", ylabel="YoY inflation (%)")
        ax5.legend(loc='upper left', frameon=True)
        ax5.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        fig5.text(0.5, 0.01, citation, ha='center', va='top', fontsize=10, wrap=True)
        fig5.tight_layout()
        fig5.savefig(os.path.join(country_out, "mich_vs_forecast.png"),
                     bbox_inches='tight', pad_inches=0.1)
        print("▸ Saved plot: mich_vs_forecast.png")
        plt.show()
        plt.close(fig5)


    # 5) Error statistics CSV (aligned to OOS period)
    # Restrict to out-of-sample period where forecasts exist
    oos_mask = merged['exp_mean'].notna()
    sub = merged.loc[oos_mask]

    stats = {
        'OOS': compute_metrics(sub['y'], sub['exp_mean']),
        'Adaptive': compute_metrics(sub['y'], sub['adaptive']),
    }
    bias_list = [
        (sub['exp_mean'] - sub['y']).mean(),
        (sub['adaptive'] - sub['y']).mean(),
    ]
    if 'mich' in sub.columns:
        mich_mask = sub['mich'].notna()
        stats['Michigan'] = compute_metrics(sub.loc[mich_mask, 'y'], sub.loc[mich_mask, 'mich'])
        bias_list.append((sub.loc[mich_mask, 'mich'] - sub.loc[mich_mask, 'y']).mean())
    stats_df = pd.DataFrame(stats).T
    stats_df['Bias'] = bias_list
    # Annotate CSV with the OOS period
    stats_df['Period'] = 'OOS'
    stats_df['PeriodStart'] = first_oos                        # use the actual first OOS forecast date
    stats_df['PeriodEnd']   = merged['exp_mean'].last_valid_index()  # last OOS forecast date
    stats_csv = os.path.join(country_out, f"error_stats_{country_code}.csv")
    stats_df.to_csv(stats_csv)
    print(f"▸ Saved error stats: error_stats_{country_code}.csv")

    # 5b) Full-period error statistics CSV
    # Compute metrics over each series’ full available history
    mask_oos_full = merged['exp_mean'].notna()
    mask_adapt_full = merged['adaptive'].notna()
    stats_full = {
        'OOS': compute_metrics(merged['y'].loc[mask_oos_full],
                               merged['exp_mean'].loc[mask_oos_full]),
        'Adaptive': compute_metrics(merged['y'].loc[mask_adapt_full],
                                    merged['adaptive'].loc[mask_adapt_full]),
    }
    bias_list_full = [
        (merged['exp_mean'] - merged['y']).dropna().mean(),
        (merged['adaptive'] - merged['y']).dropna().mean(),
    ]
    if 'mich' in merged.columns:
        mask_mich_full = merged['mich'].notna()
        stats_full['Michigan'] = compute_metrics(merged['y'].loc[mask_mich_full],
                                                 merged['mich'].loc[mask_mich_full])
        bias_list_full.append((merged['mich'] - merged['y']).dropna().mean())
    stats_full_df = pd.DataFrame(stats_full).T
    stats_full_df['Bias'] = bias_list_full
    # Annotate CSV with each series’ full-availability period
    period_starts = []
    period_ends = []
    for metric in stats_full_df.index:
        if metric == 'OOS':
            start = merged['exp_mean'].first_valid_index()
            end   = merged['exp_mean'].last_valid_index()
        elif metric == 'Adaptive':
            start = merged['adaptive'].first_valid_index()
            end   = merged['adaptive'].last_valid_index()
        elif metric == 'Michigan':
            start = merged['mich'].first_valid_index()
            end   = merged['mich'].last_valid_index()
        else:
            start = merged.index.min()
            end   = merged.index.max()
        period_starts.append(start)
        period_ends.append(end)
    stats_full_df['Period'] = 'Full'
    stats_full_df['PeriodStart'] = period_starts
    stats_full_df['PeriodEnd']   = period_ends
    stats_full_csv = os.path.join(country_out, f"error_stats_full_{country_code}.csv")
    stats_full_df.to_csv(stats_full_csv)
    print(f"▸ Saved full-period error stats: error_stats_full_{country_code}.csv")

    # 6) Forecast Errors Plot
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    # Compute errors
    err_oos = merged['exp_mean'] - merged['y']
    err_adaptive = merged['adaptive'] - merged['y']
    ax6.plot(merged.index, err_oos,      color=COL_RE_OOS,   linewidth=2, linestyle=':', label='OOS Error')
    ax6.plot(merged.index, err_adaptive, color=COL_ADAPTIVE, linewidth=2, linestyle=':', label='Adaptive Error')
    if 'mich' in merged.columns:
        err_mich = merged['mich'] - merged['y']
        ax6.plot(merged.index, err_mich, color=COL_MICHIGAN, linewidth=2, linestyle=':', label='Michigan Error')
    # Horizontal zero reference line
    ax6.axhline(0, color=COL_TRUE, linestyle=':', linewidth=1)
    ax6.set(title=f"Forecast Errors ({country_code})",
            xlabel="Date", ylabel="Error (percentage points)")
    ax6.legend(loc='upper left', frameon=True)
    ax6.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig6.tight_layout()
    fig6.savefig(os.path.join(country_out, "forecast_errors.png"),
                 bbox_inches='tight', pad_inches=0.1)
    print("▸ Saved plot: forecast_errors.png")
    plt.show()
    plt.close(fig6)
    
def process_country(country_code: str, cfg: dict, args):
    global CACHE_DIR
    """Run the full RE & adaptive pipeline for one country and save CSV."""
    series_map = cfg["series"]
    # Build regressors list: all series except 'cpi'
    regressors = [key for key in series_map if key != "cpi"]
    # Remove duplicates and keep order
    seen = set()
    regressors = [x for x in regressors if not (x in seen or seen.add(x))]
    # Patch global REGRESSORS for make_prophet and fc_regressors_random_walk
    make_prophet.REGRESSORS = regressors
    fc_regressors_random_walk.REGRESSORS = regressors
    # Prepare output directory
    country_out = os.path.join(OUTPUT_DIR, country_code)
    os.makedirs(country_out, exist_ok=True)
    # Use a separate model-cache directory for this country
    country_cache_dir = os.path.join(country_out, "_model_cache_ra")
    os.makedirs(country_cache_dir, exist_ok=True)
    CACHE_DIR = country_cache_dir
    csv_path = os.path.join(country_out, f"expectations_{country_code}.csv")

    # If cached CSV exists and not forcing, reload and regenerate plots/stats
    if os.path.exists(csv_path) and not args.force:
        print(f"▸ Loading existing expectations for {country_code} from {csv_path}")
        merged = pd.read_csv(csv_path, parse_dates=["ds"]).set_index("ds")
        if args.plot == "on":
            _generate_plots_and_stats(country_code, cfg, args, None, merged, None)
        return


    print(f"▸ Processing country {country_code}")
    # 1) Fetch data
    df = fetch_all(series_map, args.start, args.end, vintage=(args.vintage=="on"))
    # 2) Hyperparam search on first CV_YEARS
    base = df[df["ds"] <= df["ds"].min()+pd.DateOffset(years=CV_YEARS)]
    pool = bayes_search(base)
    # Save model hyperparameters for this country
    params_df = pd.DataFrame(pool, columns=["cps", "seas"])
    params_csv = os.path.join(country_out, f"model_params_{country_code}.csv")
    params_df.to_csv(params_csv, index=False)
    print(f"▸ Saved model parameters CSV for {country_code} at {params_csv}")
    # 3) OOS walk-forward
    oos = walk_forward(df, pool, force=args.force).dropna()
    # 4) Adaptive series
    adap = df.set_index("ds")["y"].shift(HORIZON_MONTHS)
    # 5) Merge y, rational, adaptive (retain full actual history)
    df_indexed = df.set_index("ds")
    merged = df_indexed[["y"]].copy()               # full history of actual inflation
    merged = merged.join(oos, how="left")           # add OOS forecasts (NaN before first forecast)
    merged["adaptive"] = adap                        # adaptive expectations (shifted actuals)
    # Add Michigan expectations only for US (if present)
    if country_code == "US" and "mich" in df.columns:
        mich_shifted = df.set_index("ds")["mich"].shift(HORIZON_MONTHS)
        merged["mich"] = mich_shifted
    # 6) Save to CSV
    # Determine columns to save
    cols = ["ds", "y", "exp_mean", "adaptive"]
    if "mich" in merged.columns:
        cols.append("mich")
    out = merged.reset_index()[cols]
    out.to_csv(csv_path, index=False)
    print(f"▸ Saved expectations CSV for {country_code} at {csv_path}")
    # Plotting and stats are now handled by _generate_plots_and_stats when CSV is cached, or here after modeling.
    if args.plot == "on":
        _generate_plots_and_stats(country_code, cfg, args, df, merged, pool)


def main():
    args = parse_args()
    for country_code, cfg in COUNTRY_CONFIG.items():
        process_country(country_code, cfg, args)

    print("▸ All countries processed.")


if __name__ == "__main__":
    main()