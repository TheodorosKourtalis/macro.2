#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 20:57:26 2025

@author: 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Nested mapping of variable names to sheet names in Quarterly_Data.xlsx.
# Εδώ ορίζουμε μόνο "Nominal" και "Deflator".
# Το Real θα το υπολογίσουμε μέσω deflator (Nominal / (Deflator/100)).
sheet_info = {
    "Gross domestic product at market prices": {
        "Nominal": "Sheet 40",  # Current prices, million euro
        "Deflator": "Sheet 79"  # Price index (implicit deflator), 2015=100, euro
    },
    "Final consumption expenditure": {
        "Nominal": "Sheet 42",  # Current prices, million euro
        "Deflator": "Sheet 81"  # Price index (implicit deflator), 2015=100, euro
    },
    "Gross fixed capital formation": {
        "Nominal": "Sheet 51",  # Current prices, million euro
        "Deflator": "Sheet 90"  # Price index (implicit deflator), 2015=100, euro
    }
}

excel_file = "Quarterly_Data.xlsx"

def clean_cell(cell):
    """
    Cleans a cell value:
      - If the cell equals ":", returns np.nan.
      - If the cell is a string containing a number with possible trailing flags (e.g., "123p", "45.6 b"),
        extracts and returns only the numeric portion as a float.
      - Otherwise, attempts to convert the cell to float.
    """
    if isinstance(cell, str):
        cell = cell.strip()
        if cell == ":":
            return np.nan
        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", cell)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return np.nan
        else:
            try:
                return float(cell)
            except Exception:
                return np.nan
    try:
        return float(cell)
    except Exception:
        return np.nan

def load_and_clean_sheet(sheet):
    """
    Loads a sheet from Quarterly_Data.xlsx and cleans the data.
    
    Assumptions:
      - Row 10 (index 9) contains quarter labels (e.g. "1995-Q1", "1995-Q2", …).
      - Row 12 (index 11) contains Euro area data.
      - Row 13 (index 12) contains Greece data.
      
    Returns:
      - A cleaned DataFrame with two columns ("Euro" and "Greece"), indexed by the quarter labels.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet, header=None)
    # Quarter labels are assumed to be in row 10 (index 9)
    time_labels = df.iloc[9].values
    try:
        start_idx = np.where(time_labels == "1995-Q1")[0][0]
    except IndexError:
        print(f"Warning: '1995-Q1' not found in sheet {sheet}. Using all columns.")
        start_idx = 0
    quarter_labels = df.iloc[9, start_idx:].values
    
    # Euro area data in row 12 (index 11)
    euro_data = df.iloc[11, start_idx:].apply(clean_cell).values
    # Greece data in row 13 (index 12)
    greece_data = df.iloc[12, start_idx:].apply(clean_cell).values
    
    cleaned_df = pd.DataFrame({
        "Euro": euro_data,
        "Greece": greece_data
    }, index=quarter_labels)
    
    # Forward/backward fill
    cleaned_df = cleaned_df.ffill().bfill()
    
    # Αφαιρούμε τυχόν σειρές που είναι εντελώς NaN ή ίδιες
    cleaned_df = cleaned_df.loc[cleaned_df.diff().abs().sum(axis=1) != 0]
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def compute_growth(df):
    """
    Computes growth rates as the first difference of the natural logarithm.
    Expects a DataFrame with numeric data (and no zeros).
    Returns a DataFrame of growth rates.
    """
    df = df.replace(0, np.nan).ffill().bfill()
    log_vals = np.log(df)
    growth = log_vals.diff().iloc[1:]
    return growth

def plot_combined_growth(x, x_labels, growth_nom, growth_real, growth_def, var_name, filename):
    """
    Creates a vertical (stacked) plot for the combined growth rates of a measure.
    
    The plot shows three lines for each region (Euro Zone on top, Greece on bottom):
      - Nominal growth
      - Real growth (calculated via Deflator)
      - Deflator growth
    
    Enhanced with larger figure size, font sizes, gridlines, and improved spacing.
    """
    fig, axs = plt.subplots(2, 1, figsize=(30, 14), dpi=150, sharex=True)
    
    # Define font sizes.
    title_font = {"fontsize": 18, "fontweight": "bold"}
    label_font = {"fontsize": 14}
    tick_font = {"fontsize": 12}
    
    # Euro Zone subplot.
    axs[0].plot(x, growth_nom.values[:, 0], marker='o', linestyle='-', linewidth=2, color='tab:blue', label="Nominal")
    axs[0].plot(x, growth_real.values[:, 0], marker='s', linestyle='-', linewidth=2, color='tab:green', label="Real")
    axs[0].plot(x, growth_def.values[:, 0], marker='^', linestyle='-', linewidth=2, color='tab:red', label="Deflator")
    axs[0].set_title(f"{var_name} - Euro Zone", **title_font)
    axs[0].set_ylabel("Growth Rate (Δ log(series))", **label_font)
    axs[0].legend(prop={'size': 12})
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Greece subplot.
    axs[1].plot(x, growth_nom.values[:, 1], marker='o', linestyle='-', linewidth=2, color='tab:blue', label="Nominal")
    axs[1].plot(x, growth_real.values[:, 1], marker='s', linestyle='-', linewidth=2, color='tab:green', label="Real")
    axs[1].plot(x, growth_def.values[:, 1], marker='^', linestyle='-', linewidth=2, color='tab:red', label="Deflator")
    axs[1].set_title(f"{var_name} - Greece", **title_font)
    axs[1].set_ylabel("Growth Rate (Δ log(series))", **label_font)
    axs[1].set_xlabel("Quarter", **label_font)
    axs[1].legend(prop={'size': 12})
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Show every 10th tick label (adjust as necessary).
    step = 10
    xtick_indices = x[::step]
    xtick_labels = x_labels[::step]
    axs[1].set_xticks(xtick_indices)
    axs[1].set_xticklabels(xtick_labels, rotation=45, ha='right', **tick_font)
    
    # Adjust tick parameters.
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    
    fig.suptitle(f"{var_name} Growth Rates (Quarterly)", fontsize=20, fontweight="bold")
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95, hspace=0.35)
    plt.savefig(filename)
    plt.close()
    print(f"Combined growth plot for '{var_name}' saved as: {filename}")

def main():
    data_dict = {}
    growth_dict = {}
    
    for measure, sheets in sheet_info.items():
        print(f"Processing measure '{measure}':")
        try:
            # 1. Φόρτωση Nominal
            df_nom = load_and_clean_sheet(sheets["Nominal"])
            # 2. Φόρτωση Deflator
            df_def = load_and_clean_sheet(sheets["Deflator"])
            
            # Ελέγχουμε αν έχουν κοινό χρονικό εύρος
            common_index = df_nom.index.intersection(df_def.index)
            if common_index.empty:
                print(f"No common quarter labels for '{measure}'. Skipping measure.")
                continue
            
            df_nom = df_nom.loc[common_index]
            df_def = df_def.loc[common_index]
            
            # 3. Υπολογισμός Real = Nominal / (Deflator/100)
            df_real = df_nom / (df_def / 100)
            
            # Αποθηκεύουμε τα DataFrame
            data_dict[measure] = {
                "Nominal": df_nom,
                "Deflator": df_def,
                "Real": df_real
            }
            
            # 4. Υπολογίζουμε growth rates για Nominal, Real, Deflator
            growth_nom = compute_growth(df_nom)
            growth_def = compute_growth(df_def)
            growth_real = compute_growth(df_real)
            
            growth_dict[measure] = {
                "Nominal": growth_nom,
                "Deflator": growth_def,
                "Real": growth_real
            }
            
            # 5. Plot
            x_labels = df_nom.index[1:]
            x = np.arange(len(x_labels))
            
            plot_filename = measure.replace(" ", "_").replace("/", "_") + "_combined_growth.png"
            plot_combined_growth(x, x_labels, growth_nom, growth_real, growth_def, measure, plot_filename)
            
        except Exception as e:
            print(f"Error processing '{measure}': {e}")
    
    print("All combined growth plots have been generated.")

if __name__ == "__main__":
    main()