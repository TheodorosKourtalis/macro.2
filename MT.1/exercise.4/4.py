#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:47:30 2025

@author: thodoreskourtales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# New mapping: for each measure, specify the Excel sheet for Nominal and for Chain linked (real).
sheet_info = {
    "Nominal GDP": {
        "Nominal": "Sheet 40",          # Current prices, million euro (Gross domestic product at market prices)
        "Chain linked": "Sheet 79"       # Chain linked volumes, index 2015=100 (Gross domestic product at market prices)
    },
    "Final consumption expenditure": {
        "Nominal": "Sheet 42",           # Current prices, million euro (Final consumption expenditure)
        "Chain linked": "Sheet 81"       # Chain linked volumes, index 2015=100 (Final consumption expenditure)
    },
    "Final consumption expenditure of general government": {
        "Nominal": "Sheet 43",           # Current prices, million euro (Final consumption expenditure of general government)
        "Chain linked": "Sheet 82"       # Chain linked volumes, index 2015=100 (Final consumption expenditure of general government)
    },
    "Final consumption expenditure of households": {
        "Nominal": "Sheet 47",           # Current prices, million euro (Final consumption expenditure of households)
        "Chain linked": "Sheet 86"       # Chain linked volumes, index 2015=100 (Final consumption expenditure of households)
    },
    "Gross fixed capital formation": {
        "Nominal": "Sheet 51",           # Current prices, million euro (Gross fixed capital formation)
        "Chain linked": "Sheet 90"       # Chain linked volumes, index 2015=100 (Gross fixed capital formation)
    },
    "Exports of goods and services": {
        "Nominal": "Sheet 55",           # Current prices, million euro (Exports of goods and services)
        "Chain linked": "Sheet 94"       # Chain linked volumes, index 2015=100 (Exports of goods and services)
    },
    "Imports of goods and services": {
        "Nominal": "Sheet 58",           # Current prices, million euro (Imports of goods and services)
        "Chain linked": "Sheet 97"       # Chain linked volumes, index 2015=100 (Imports of goods and services)
    }
}
# Dictionary to store summary info for each measure.
report_data = {}

def convert_to_float(x):
    """Convert a cell value to float; treat ':' as NaN."""
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == ":":
                return np.nan
        return float(x)
    except:
        return np.nan

def plot_growth_side_by_side(years, growth_values, var_name, filename, scale_threshold=2):
    """
    Create a side-by-side growth plot for two regions (assumed first column = Euro Zone,
    second column = Greece).
    """
    std_first = np.nanstd(growth_values[:, 0])
    std_second = np.nanstd(growth_values[:, 1])
    if std_second > scale_threshold * std_first:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    axs[0].plot(years, growth_values[:, 0], marker='o', linestyle='-', linewidth=2, color='tab:blue', label=var_name)
    axs[0].set_title('Euro Zone')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Growth Rate (Δ log(series))')
    axs[0].legend()
    axs[1].plot(years, growth_values[:, 1], marker='o', linestyle='-', linewidth=2, color='tab:orange', label=var_name)
    axs[1].set_title('Greece')
    axs[1].set_xlabel('Year')
    axs[1].legend()
    fig.suptitle(f'{var_name} Growth Rates (1995–2022)', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Growth plot for '{var_name}' saved as: {filename}")

def plot_measure_combined(measure_name, years, nominal, chain, deflator, base_year_target, filename):
    """
    Create a combined plot for levels of a measure (Nominal, Chain linked, Deflator)
    rebased to base_year_target.
    """
    idx = np.where(years == base_year_target)[0]
    idx = idx[0] if len(idx) > 0 else 0
    nominal_norm = nominal / nominal[idx, :] * 100
    chain_norm = chain / chain[idx, :] * 100
    deflator_norm = deflator / deflator[idx, :] * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
    axs[0].plot(years, nominal_norm[:, 0], marker='o', linestyle='-', label="Nominal")
    axs[0].plot(years, chain_norm[:, 0], marker='s', linestyle='-', label="Chain linked")
    axs[0].plot(years, deflator_norm[:, 0], marker='^', linestyle='-', label="Deflator")
    axs[0].set_title("Euro Zone")
    axs[0].set_xlabel("Year")
    axs[0].set_ylabel(f"Index (Base {base_year_target} = 100)")
    axs[0].legend()
    axs[1].plot(years, nominal_norm[:, 1], marker='o', linestyle='-', label="Nominal")
    axs[1].plot(years, chain_norm[:, 1], marker='s', linestyle='-', label="Chain linked")
    axs[1].plot(years, deflator_norm[:, 1], marker='^', linestyle='-', label="Deflator")
    axs[1].set_title("Greece")
    axs[1].set_xlabel("Year")
    axs[1].legend()
    fig.suptitle(f"{measure_name} Levels (Rebased to {base_year_target} = 100)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Combined levels plot for '{measure_name}' saved as: {filename}")

def plot_measure_growth_by_country(measure_name, years, growth_nominal, growth_chain, growth_deflator, filename_euro, filename_greece):
    """
    Create separate combined growth plots for a measure (Nominal, Chain linked, Deflator)
    for Euro Zone and Greece.
    """
    # Euro Zone plot.
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(years, growth_nominal[:, 0], marker='o', linestyle='-', label=f"{measure_name} Nominal Growth")
    plt.plot(years, growth_chain[:, 0], marker='s', linestyle='-', label=f"{measure_name} Chain Growth")
    plt.plot(years, growth_deflator[:, 0], marker='^', linestyle='-', label=f"{measure_name} Deflator Growth")
    plt.title(f"Combined Growth Rates ({measure_name}) – Euro Zone")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (Δ log(series))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename_euro)
    plt.close()
    print(f"Combined growth plot for '{measure_name}' (Euro Zone) saved as: {filename_euro}")
    
    # Greece plot.
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(years, growth_nominal[:, 1], marker='o', linestyle='-', label=f"{measure_name} Nominal Growth")
    plt.plot(years, growth_chain[:, 1], marker='s', linestyle='-', label=f"{measure_name} Chain Growth")
    plt.plot(years, growth_deflator[:, 1], marker='^', linestyle='-', label=f"{measure_name} Deflator Growth")
    plt.title(f"Combined Growth Rates ({measure_name}) – Greece")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (Δ log(series))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename_greece)
    plt.close()
    print(f"Combined growth plot for '{measure_name}' (Greece) saved as: {filename_greece}")

def main():
    # Process each measure in the new sheet_info.
    for measure_name, sheets in sheet_info.items():
        try:
            # Read both the Nominal and Chain linked sheets.
            df_nom = pd.read_excel("Annual_Data.xlsx", sheet_name=sheets["Nominal"], header=None, skiprows=8, nrows=4)
            df_chain = pd.read_excel("Annual_Data.xlsx", sheet_name=sheets["Chain linked"], header=None, skiprows=8, nrows=4)
        except Exception as e:
            print(f"Error reading sheets for {measure_name}: {e}")
            continue

        print(f"Processing measure '{measure_name}' with Nominal sheet '{sheets['Nominal']}' and Chain linked sheet '{sheets['Chain linked']}'")
        
        # Get the time row (assumed identical in both sheets).
        time_row = pd.to_numeric(df_nom.iloc[0].values, errors="coerce")
        selected_cols = np.where((time_row >= 1995) & (time_row <= 2022))[0]
        if len(selected_cols) == 0:
            print(f"Warning: No columns for 1995–2022 in measure '{measure_name}'.")
            continue
        
        # Extract the data for Euro Zone and Greece.
        df_nom_data = df_nom.iloc[2:4, selected_cols]
        df_chain_data = df_chain.iloc[2:4, selected_cols]
        df_nom_data = df_nom_data.applymap(convert_to_float).ffill(axis=1)
        df_chain_data = df_chain_data.applymap(convert_to_float).ffill(axis=1)
        
        # Convert to NumPy arrays.
        values_nom = df_nom_data.transpose().values
        values_chain = df_chain_data.transpose().values
        if np.all(np.isnan(values_nom)) or np.all(np.isnan(values_chain)):
            print(f"Warning: Data conversion issue for {measure_name}.")
            continue
        
        # Calculate the deflator for the measure.
        deflator = (values_nom / values_chain) * 100
        
        # Compute logarithms and growth rates.
        log_nom = np.log(values_nom)
        log_chain = np.log(values_chain)
        log_def = np.log(deflator)
        growth_nom = np.diff(log_nom, axis=0)
        growth_chain = np.diff(log_chain, axis=0)
        growth_def = np.diff(log_def, axis=0)
        
        # Extract the time points.
        selected_years = time_row[selected_cols]
        growth_years = selected_years[1:]
        
        # Plot growth for each series.
        filename_nom_growth = measure_name.replace(" ", "_").replace("/", "_") + "_nominal_growth.png"
        plot_growth_side_by_side(growth_years, growth_nom, measure_name + " Nominal", filename_nom_growth)
        
        filename_chain_growth = measure_name.replace(" ", "_").replace("/", "_") + "_chain_growth.png"
        plot_growth_side_by_side(growth_years, growth_chain, measure_name + " Chain linked", filename_chain_growth)
        
        filename_deflator_growth = measure_name.replace(" ", "_").replace("/", "_") + "_deflator_growth.png"
        plot_growth_side_by_side(growth_years, growth_def, measure_name + " Deflator", filename_deflator_growth)
        
        # Combined levels plot.
        base_year_target = 2015
        filename_combined_levels = measure_name.replace(" ", "_").replace("/", "_") + "_combined_levels.png"
        plot_measure_combined(measure_name, selected_years, values_nom, values_chain, deflator, base_year_target, filename_combined_levels)
        
        # Combined growth plot by country.
        filename_combined_growth_euro = measure_name.replace(" ", "_").replace("/", "_") + "_combined_growth_Euro.png"
        filename_combined_growth_greece = measure_name.replace(" ", "_").replace("/", "_") + "_combined_growth_Greece.png"
        plot_measure_growth_by_country(measure_name, growth_years, growth_nom, growth_chain, growth_def, 
                                       filename_combined_growth_euro, filename_combined_growth_greece)
        
        # Store summary info (this is printed at the end).
        report_data[measure_name] = {
            "num_points": values_nom.shape[0],
            "nominal_mean_growth": np.nanmean(growth_nom[:, 0]),
            "chain_mean_growth": np.nanmean(growth_chain[:, 0]),
            "deflator_mean_growth": np.nanmean(growth_def[:, 0]),
            "plot_files": {
                "nominal_growth": filename_nom_growth,
                "chain_growth": filename_chain_growth,
                "deflator_growth": filename_deflator_growth,
                "combined_levels": filename_combined_levels,
                "combined_growth_Euro": filename_combined_growth_euro,
                "combined_growth_Greece": filename_combined_growth_greece
            }
        }
    
    # Print summary information.
    print("\nProcessing completed. Summary of measures:")
    for measure, info in report_data.items():
        print(f"\n{measure}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()