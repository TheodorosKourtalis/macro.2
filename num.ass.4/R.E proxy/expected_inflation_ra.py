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
import seaborn as sns
sns.set_theme(style="whitegrid")
from matplotlib.lines import Line2D
CB_PAL = sns.color_palette("tab10", 4)

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

# Configuration
CSV_PATH = "/Users/thodoreskourtales/macro.num_ass_4/expected_inflation.csv"
SERIES = {"cpi": "CPIAUCSL", "core_cpi": "CPILFESL", "unemp_rate": "UNRATE", "fed_funds": "FEDFUNDS", "oil_price": "DCOILWTICO", "m2": "M2SL", "ppi": "PPIACO", "sentiment": "UMCSENT", "mich_exp": "MICH"}
REGRESSORS = [k for k in SERIES if k != "cpi"]
CONFINT = 0.8
CV_YEARS = 10
MAX_MODELS = 4
BO_INIT_POINTS = 5
BO_N_ITER = 20
ROLL_ERR_WIN = 36
CACHE_DIR = os.path.join(os.path.dirname(__file__), "_model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MICHIGAN_MAIZE = "#FFCB05"

# Data helpers

def fetch_series(code: str, start: str, end: str, vintage: bool) -> pd.Series:
    src = "alfred" if vintage else "fred"
    df = web.DataReader(code, src, start, end)
    ser = df.droplevel(0, axis=1).iloc[:, 0] if vintage else df.iloc[:, 0]
    return ser.resample("MS").last()


def fetch_all(start: str, end: str, vintage: bool=False) -> pd.DataFrame:
    data = {}
    for name, code in SERIES.items():
        try:
            data[name] = fetch_series(code, start, end, vintage)
        except Exception as e:
            print(f"⚠️ failed to fetch {name}: {e}")
    if "cpi" not in data:
        raise ValueError("CPI missing after fetch.")
    idx = data["cpi"].index
    df = pd.DataFrame({n: s.reindex(idx) for n, s in data.items()})
    df["y"] = df["cpi"].pct_change(12) * 100
    df[REGRESSORS] = df[REGRESSORS].ffill().bfill()
    df = df.dropna(subset=["y"]).reset_index()
    df.rename(columns={df.columns[0]: "ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
    return df[["ds", "y"] + REGRESSORS]

# Prophet factory

def make_prophet(cps: float, seas: float) -> Prophet:
    m = Prophet(growth="linear", interval_width=CONFINT, yearly_seasonality=True,
                weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=cps, seasonality_prior_scale=seas)
    m.add_country_holidays(country_name="US")
    m.add_seasonality("quarterly", period=91.3, fourier_order=5)
    for r in REGRESSORS:
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

# Random-walk forecast

def fc_regressors_random_walk(hist: pd.DataFrame) -> pd.DataFrame:
    next_ds = hist["ds"].max() + pd.offsets.MonthBegin()
    vals = hist.iloc[-1][REGRESSORS]
    return pd.DataFrame({"ds":[next_ds], **vals.to_dict()})

# Metrics

def compute_metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = (true - pred).abs()
    return {"MAPE": (err/true.abs()).mean()*100, "RMSE": np.sqrt(((true-pred)**2).mean())}

# Walk-forward

def walk_forward(df: pd.DataFrame, params_pool: list[tuple[float,float]]) -> pd.DataFrame:
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
            if os.path.exists(path): m = joblib.load(path)
            else:
                m = make_prophet(cps, seas); m.fit(hist); joblib.dump(m, path)
            yhat = m.predict(future)["yhat"].iloc[0]
            err = (hist["y"] - m.predict(hist)["yhat"]).abs()
            preds.append(yhat); weights.append(1/err.tail(ROLL_ERR_WIN).mean())
        recs.append({"ds":future["ds"].iloc[0],"exp_mean":np.average(preds,weights=weights),**{f"model_{i+1}":p for i,p in enumerate(preds)}})
    return pd.DataFrame(recs).set_index("ds")

# CLI / main

def parse_args():
    today = dt.date.today().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser("Expected inflation RE proxy")
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--end", default=today)
    p.add_argument("--vintage", choices=["on","off"], default="off")
    p.add_argument("--plot", choices=["on","off"], default="on")
    return p.parse_args()

def main():
    args = parse_args()
    existing_csv = os.path.exists(CSV_PATH)

    if existing_csv:
        print(f"▸ Loading cached expectations from {CSV_PATH}")
        spec_lines = []
        with open(CSV_PATH, "r") as fh:
            for line in fh:
                if not line.startswith("#"): break
                spec_lines.append(line.lstrip("# ").rstrip())
        for l in spec_lines:
            if l.startswith("model_"): print("  ✓", l)
        oos = pd.read_csv(CSV_PATH, comment='#', index_col=0, parse_dates=True)
        # Always fetch full history as specified by args
        df = fetch_all(args.start, args.end, vintage=(args.vintage=="on"))
    else:
        print("▸ Fetching data …")
        df = fetch_all(args.start, args.end, vintage=(args.vintage=="on"))
        print("▸ Hyperparam search …")
        base = df[df["ds"]<=df["ds"].min()+pd.DateOffset(years=CV_YEARS)]
        pool = bayes_search(base)
        spec_lines = [f"model_{i}: cps={cps:.3f},seas={seas:.3f}" for i,(cps,seas) in enumerate(pool,1)]
        for l in spec_lines: print("  ✓", l)
        print("▸ OOS walk-forward …")
        oos = walk_forward(df,pool)
        with open(CSV_PATH, "w") as fh:
            for l in spec_lines: fh.write(f"# {l}\n")
            oos.to_csv(fh)
        print(f"▸ Saved {CSV_PATH}")

    merged = oos.join(df.set_index("ds")["y"], how="left").dropna()
    m_oos = compute_metrics(merged["y"], merged["exp_mean"])
    print(f"▸ OOS metrics — MAPE: {m_oos['MAPE']:.2f}%, RMSE: {m_oos['RMSE']:.3f}")

    print("▸ In-sample ensemble …")
    if existing_csv:
        pool = []
        for l in spec_lines:
            if l.startswith("model_"):
                parts = l.split(":")[-1].split(",")
                pool.append((float(parts[0].split("=")[1]), float(parts[1].split("=")[1])))
    in_df = pd.DataFrame({'ds':df['ds']})
    weights=[]

    for i,(cps,seas) in enumerate(pool,1):
        print(f"  • fitting model_{i}")
        m = make_prophet(cps,seas); m.fit(df)
        yhat = m.predict(df)["yhat"]; in_df[f"model_{i}"] = yhat
        err = (df["y"] - yhat).abs(); weights.append(1/err.tail(ROLL_ERR_WIN).mean())
    # ensemble mean with weights; fall back to equal weights if sum invalid
    models = [f"model_{i}" for i in range(1, len(pool)+1)]
    weight_sum = sum(weights)
    if not np.isfinite(weight_sum) or weight_sum == 0:
        print("⚠️ weights sum to zero or invalid; using equal weights for ensemble")
        in_df['exp_mean'] = in_df[models].mean(axis=1)
    else:
        in_df['exp_mean'] = np.average(in_df[models], axis=1, weights=weights)
    m_in = compute_metrics(df['y'], in_df['exp_mean'])
    print(f"▸ In-sample metrics — MAPE: {m_in['MAPE']:.2f}%, RMSE: {m_in['RMSE']:.3f}")
    adap_series = df.set_index("ds")["y"].shift(1)
    
    mich_series = df.set_index("ds")["mich_exp"]

    box_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, ec="0.5")

    # OOS plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Highlight forecast start and recessions
    forecast_start = merged.index.min()
    ax.axvline(forecast_start, color='black', linestyle='--', linewidth=1)
    ax.axvspan(forecast_start, df['ds'].max(), color='lightgrey', alpha=0.3)
    # Shade NBER recessions
    rec = web.DataReader('USREC', 'fred', df['ds'].min(), df['ds'].max())['USREC']
    rec = rec.resample('MS').last()
    for date, val in rec.items():
        if val == 1:
            ax.axvspan(date, date + pd.offsets.MonthEnd(), color='gray', alpha=0.2, linewidth=0)
    ax.plot(df.set_index('ds')['y'], linestyle='-', linewidth=2, color='black', label='Actual π')
    ax.plot(
        merged['exp_mean'],
        linestyle=':', linewidth=2, color='green', label='Rational Exp π (OOS)'
    )
    ax.plot(
        adap_series,
        linestyle=':', linewidth=2, color='gray', label='Adapt. Exp π'
    )
    ax.plot(
        mich_series,
        linestyle=':', linewidth=2, color=MICHIGAN_MAIZE, label='Survey π (MICH)'
    )
    ax.set(title="Actual vs All Proxies", xlabel="Date", ylabel="YoY inflation (%)")
    ax.legend(loc='upper left', frameon=True)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    sources_text = (
        "Sources: BLS CPIAUCSL; Univ. of Michigan MICH; FRB St. Louis FRED/ALFRED; Theodoros Kourtalis. "
        "Adaptive Expectations = lag-1. Rational expectations=Finetuned FB-Prophet"
    )
    fig.subplots_adjust(bottom=0.45)
    fig.text(0.5, 0.01, sources_text, ha='center', va='top', fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(os.path.abspath("actual_vs_all_proxies.png"), bbox_inches='tight', pad_inches=0.1)
    print("▸ Saved actual_vs_all_proxies.png")

    # In-sample plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df.set_index('ds')['y'], linestyle='-', linewidth=2, color='black', label='Actual π')
    ax2.plot(
        in_df.set_index('ds')['exp_mean'],
        linestyle=':', linewidth=2, color='blue', label='Rational Exp π (IS)'
    )
    ax2.set(title="Actual vs Expected Inflation — In-Sample", xlabel="Date", ylabel="YoY inflation (%)")
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    fig2.subplots_adjust(bottom=0.45)
    fig2.text(0.5, 0.01, sources_text, ha='center', va='top', fontsize=10, wrap=True)
    fig2.tight_layout()
    fig2.savefig(os.path.abspath("actual_vs_Rational Exp (IN).png"), bbox_inches='tight', pad_inches=0.1)
    print("▸ Saved actual_vs_Rational_Exp(IN).png")

    # ------------------------------------------------------------------ Combined Forecast & Michigan Plot
    # Compute Michigan OOS metrics
    m_mich_oos = compute_metrics(merged['y'], mich_series.loc[merged.index])
    # In-sample metrics already in m_in, OOS in m_oos
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df.set_index('ds')['y'], linestyle='-', linewidth=2, color='black', label='Actual π')
    # Only OOS and Michigan lines (in-sample forecast line removed)
    ax3.plot(
        merged['exp_mean'],
        linestyle=':', linewidth=2, color='green', label='Rational Exp π (OOS)'
    )
    ax3.plot(
        mich_series,
        linestyle=':', linewidth=2, color=MICHIGAN_MAIZE, label='Survey π (MICH)'
    )
    ax3.set(title="Rational Exp (OOS) vs Michigan survey", xlabel="Date", ylabel="YoY inflation (%)")
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig3.subplots_adjust(bottom=0.50)
    fig3.text(0.5, 0.01, sources_text, ha='center', va='top', fontsize=10, wrap=True)
    fig3.tight_layout()
    fig3.savefig(os.path.abspath('Rational_Exp(OOS)_vs_mich.png'), bbox_inches='tight', pad_inches=0.1)
    print('▸ Saved Rational_Exp(OOS)_vs_mich.png')

    # ------------------------------------------------------------------ Adaptive vs Forecast Plot
    # Compute Adaptive OOS metrics
    m_adap_oos = compute_metrics(merged['y'], adap_series.loc[merged.index])
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(df.set_index('ds')['y'], linestyle='-', linewidth=2, color='black', label='Actual π')
    ax4.plot(
        adap_series,
        linestyle=':', linewidth=2, color='gray', label='Adapt. Exp π'
    )
    ax4.plot(
        merged['exp_mean'],
        linestyle=':', linewidth=2, color='green', label='Rational Exp π (OOS)'
    )
    ax4.set(title="Adaptive vs Rational Exp(OOS)", xlabel="Date", ylabel="YoY inflation (%)")
    ax4.legend(loc='upper left', frameon=True)
    ax4.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    fig4.subplots_adjust(bottom=0.50)
    fig4.text(0.5, 0.01, sources_text, ha='center', va='top', fontsize=10, wrap=True)
    fig4.tight_layout()
    fig4.savefig(os.path.abspath('adaptive_vs_Rational_Exp(OOS).png'), bbox_inches='tight', pad_inches=0.1)
    print('▸ Saved adaptive_vs_Rational_Exp(OOS).png')



    # ------------------------------------------------------------------ Export error statistics
    stats = {}
    # In-sample forecast errors
    stats['EXP-IS'] = compute_metrics(df['y'], in_df['exp_mean'])
    # OOS forecast errors
    stats['OOS'] = compute_metrics(merged['y'], merged['exp_mean'])
    # Adaptive expectations errors
    stats['Adapt.EXP(LAG-1)'] = compute_metrics(merged['y'], adap_series.loc[merged.index])
    # Michigan survey errors
    stats['Survey π (MICH)'] = compute_metrics(merged['y'], mich_series.loc[merged.index])

    # Build DataFrame and add bias
    stats_df = pd.DataFrame(stats).T
    stats_df['Bias'] = [
        (in_df['exp_mean'] - df['y']).mean(),
        (merged['exp_mean'] - merged['y']).mean(),
        (adap_series.loc[merged.index] - merged['y']).mean(),
        (mich_series.loc[merged.index] - merged['y']).mean()
    ]

    # Save to CSV
    stats_csv = os.path.abspath('error_stats.csv')
    stats_df.to_csv(stats_csv)
    print(f"▸ Saved error statistics to {stats_csv}")

    # ------------------------------------------------------------------ Difference from Actual π (Small Multiples)
    # Compute errors relative to true inflation
    diff_is  = in_df.set_index('ds')['exp_mean'] - df.set_index('ds')['y']
    diff_oos = merged['exp_mean'] - merged['y']
    diff_ad  = adap_series - df.set_index('ds')['y']
    diff_mi  = mich_series - df.set_index('ds')['y']

    fig5, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    series = [
        ('Rational Exp π (IS) Error', diff_is, 'blue'),
        ('Rational Exp π (OOS) Error', diff_oos, 'green'),
        ('Adapt. Exp π Error',       diff_ad, 'gray'),
        ('Survey π Error',           diff_mi, MICHIGAN_MAIZE)
    ]

    for ax, (label, data, color) in zip(axes, series):
        ax.plot(data, linewidth=2, linestyle='-', color=color)
        ax.set_title(label)
        ax.set_ylabel('Error (pp)')
        ax.grid(True)

    axes[-1].set_xlabel('Date')
    fig5.tight_layout()
    fig5.subplots_adjust(bottom=0.05)
    fig5.text(0.5, 0.0001, sources_text, ha='center', va='top', fontsize=10, wrap=True)
    fig5.savefig(os.path.abspath('forecast_errors.png'), bbox_inches='tight', pad_inches=0.1)
    print('▸ Saved forecast_errors.png')

if __name__ == "__main__":
    main()
