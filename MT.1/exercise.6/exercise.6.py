#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Άσκηση 6: Υπολογισμός Τάσης και Κυκλικής Συνιστώσας με το φίλτρο HP για Ευρωζώνη και Ελλάδα

Μεταβλητές (σε πραγματικούς όρους):
    - ΑΕΠ (από Value added, gross – Sheet 2)
    - Ιδιωτική Κατανάλωση (από Final consumption expenditure – Sheet 3)
    - Επενδύσεις (από Gross fixed capital formation – Sheet 12)

Δομή του Excel:
  - Σειρά 10 (index 9): Περιέχει τις ετικέτες των περιόδων (π.χ., "1995-Q1", "1995-Q2", …).  
    Τα δεδομένα ξεκινούν από το "1995-Q1" και αγνοούνται οι τιμές πριν από αυτήν.
  - Σειρά 12 (index 11): Δεδομένα για την Ευρωζώνη.
  - Σειρά 13 (index 12): Δεδομένα για την Ελλάδα.

Για κάθε μεταβλητή το script:
  1. Διαβάζει και καθαρίζει τα δεδομένα (για Euro και Ελλάδα) χρησιμοποιώντας τις παραπάνω σειρές.
  2. Εφαρμόζει το φίλτρο HP (λ = 1600) για εξαγωγή τάσης και κυκλικής συνιστώσας.
  3. Σχεδιάζει σε ένα διάγραμμα (2 υποπλοτ: πάνω για Euro, κάτω για Ελλάδα) την πραγματική τιμή και την τάση, με ετικέτες του άξονα Χ κάθε 20 περίοδοι.
  4. Σχεδιάζει σε ένα διάγραμμα (2 υποπλοτ) τις κυκλικές συνιστώσες για κάθε μεταβλητή.
  5. Σχεδιάζει συγκριτικά (χωριστά για Euro και για Ελλάδα) όλα τα διαγράμματα των κυκλικών συνιστωσών.
  6. Υπολογίζει τη μεταβλητότητα (τυπική απόκλιση) της κυκλικής συνιστώσας και τις σχετικές μεταβλητότητες (σε σχέση με το ΑΕΠ).
  7. Εξάγει πίνακες (DataFrame) με τις σχετικές μεταβλητότητες για Euro και για Ελλάδα.

Συντάκτης: thodoreskourtales
Δημιουργήθηκε: Fri Mar  7 22:23:36 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.filters.hp_filter import hpfilter
import seaborn as sns

# Ορισμός επαγγελματικού στυλ διαγραμμάτων
sns.set_style('whitegrid')

# Χαρτογράφηση μεταβλητών για την Άσκηση 6
sheet_names_ex6 = {
    "ΑΕΠ": "Sheet 2",                    # Value added, gross
    "Ιδιωτική Κατανάλωση": "Sheet 8",      # Final consumption expenditure
    "Επενδύσεις": "Sheet 12"              # Gross fixed capital formation
}

excel_file = "Quarterly_Data.xlsx"

# ----------------------------
# Συναρτήσεις Φόρτωσης & Καθαρισμού Δεδομένων
# ----------------------------

def clean_cell(cell):
    """
    Καθαρίζει μια τιμή κελιού:
      - Επιστρέφει np.nan εάν το κελί είναι ":".
      - Αντικαθιστά το κόμμα με τελεία και εξάγει το αριθμητικό μέρος.
      - Προσπαθεί να μετατρέψει την τιμή σε float.
    """
    if isinstance(cell, str):
        cell = cell.strip().replace(",", ".")
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
    Διαβάζει ένα φύλλο από το αρχείο Quarterly_Data.xlsx και καθαρίζει τα δεδομένα.
    
    Δεδομένα:
      - Η σειρά 10 (index 9) περιέχει τις ετικέτες των περιόδων (π.χ., "1995-Q1", "1995-Q2", …).
        Τα δεδομένα ξεκινούν από το "1995-Q1" και αγνοούνται οι στήλες πριν από αυτήν.
      - Η σειρά 12 (index 11) περιέχει τα δεδομένα για την Ευρωζώνη.
      - Η σειρά 13 (index 12) περιέχει τα δεδομένα για την Ελλάδα.
    
    Επιστρέφει ένα DataFrame με στήλες "Euro" και "Ελλάδα", με δείκτη τις ετικέτες των περιόδων.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet, header=None)
    # Λήψη των ετικετών περιόδων από τη σειρά 10 (index 9)
    all_labels = df.iloc[9].values
    # Βρίσκουμε τη θέση του "1995-Q1"
    try:
        start_idx = list(all_labels).index("1995-Q1")
    except ValueError:
        print("Δεν βρέθηκε το '1995-Q1', χρησιμοποιούνται όλες οι στήλες.")
        start_idx = 0
    # Επιλογή στηλών από το '1995-Q1' και μετά
    valid_cols = [i for i in range(start_idx, len(all_labels)) if pd.notna(all_labels[i])]
    # Δημιουργία λίστας ετικετών από τις έγκυρες στήλες
    quarter_labels = [str(all_labels[i]) for i in valid_cols]
    # Εξαγωγή δεδομένων: row 12 (index 11) για Euro, row 13 (index 12) για Ελλάδα
    euro_data = df.iloc[11, valid_cols].apply(clean_cell).values
    greece_data = df.iloc[12, valid_cols].apply(clean_cell).values
    cleaned_df = pd.DataFrame({"Euro": euro_data, "Ελλάδα": greece_data}, index=quarter_labels)
    cleaned_df = cleaned_df.ffill(axis=1).bfill(axis=1)
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

# ----------------------------
# Συναρτήσεις για το φίλτρο HP & Διαγράμματα
# ----------------------------

def compute_hp_decomposition(series, lamb=1600):
    """
    Εφαρμόζει το φίλτρο HP σε μία σειρά.
    Επιστρέφει:
      - cycle: την κυκλική συνιστώσα
      - trend: την τάση
    """
    cycle, trend = hpfilter(series, lamb=lamb)
    return cycle, trend

def plot_actual_vs_trend_dual(df, var_name, filename):
    """
    Σχεδιάζει σε ένα διάγραμμα με 2 υποπλοτ (πάνω: Euro, κάτω: Ελλάδα) την πραγματική τιμή και την τάση.
    Εμφανίζει τις ετικέτες του άξονα Χ κάθε 20 περίοδοι.
    """
    cycle_euro, trend_euro = compute_hp_decomposition(df["Euro"], lamb=1600)
    cycle_gr, trend_gr = compute_hp_decomposition(df["Ελλάδα"], lamb=1600)
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs[0].plot(df.index, df["Euro"], label="Πραγματική (Euro)", marker='o', linewidth=2)
    axs[0].plot(trend_euro.index, trend_euro, label="Τάση (Euro)", linestyle='--', linewidth=2, color='red')
    axs[0].set_title(f"{var_name} - Ευρωζώνη", fontsize=14)
    axs[0].legend(fontsize=12)
    
    axs[1].plot(df.index, df["Ελλάδα"], label="Πραγματική (Ελλάδα)", marker='o', linewidth=2)
    axs[1].plot(trend_gr.index, trend_gr, label="Τάση (Ελλάδα)", linestyle='--', linewidth=2, color='red')
    axs[1].set_title(f"{var_name} - Ελλάδα", fontsize=14)
    axs[1].legend(fontsize=12)
    
    xticks = df.index[::20]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks, rotation=45)
    
    fig.suptitle(f"{var_name}: Πραγματική Τιμή και Τάση", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το διάγραμμα (Πραγματική & Τάση) για {var_name} ως: {filename}")

def plot_cyclical_dual(df, var_name, filename):
    """
    Σχεδιάζει σε ένα διάγραμμα με 2 υποπλοτ την κυκλική συνιστώσα (πάνω: Euro, κάτω: Ελλάδα) για τη μεταβλητή.
    Εμφανίζει τις ετικέτες του άξονα Χ κάθε 20 περίοδοι.
    """
    cycle_euro, _ = compute_hp_decomposition(df["Euro"], lamb=1600)
    cycle_gr, _ = compute_hp_decomposition(df["Ελλάδα"], lamb=1600)
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs[0].plot(cycle_euro.index, cycle_euro, label="Κυκλική (Euro)", marker='o', linewidth=2)
    axs[0].set_title(f"{var_name} - Κυκλική (Ευρωζώνη)", fontsize=14)
    axs[0].legend(fontsize=12)
    
    axs[1].plot(cycle_gr.index, cycle_gr, label="Κυκλική (Ελλάδα)", marker='o', linewidth=2)
    axs[1].set_title(f"{var_name} - Κυκλική (Ελλάδα)", fontsize=14)
    axs[1].legend(fontsize=12)
    
    xticks = df.index[::20]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks, rotation=45)
    
    fig.suptitle(f"{var_name}: Κυκλική Συνιστώσα", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το διάγραμμα (Κυκλική) για {var_name} ως: {filename}")

def plot_all_cyclical_components_dual(cycles_dict, region, filename):
    """
    Σχεδιάζει όλες τις κυκλικές συνιστώσες για μια περιοχή (region: 'Euro' ή 'Ελλάδα') σε ένα ενιαίο διάγραμμα.
    Εμφανίζει τις ετικέτες του άξονα Χ κάθε 20 περίοδοι.
    """
    plt.figure(figsize=(14, 7))
    for var, cycles in cycles_dict.items():
        plt.plot(cycles[region].index, cycles[region], marker='o', linewidth=2, label=var)
    plt.title(f"Όλες οι Κυκλικές Συνιστώσες ({region})", fontsize=16)
    plt.xlabel("Περίοδος", fontsize=12)
    plt.ylabel("Κύκλος", fontsize=12)
    plt.legend(fontsize=12)
    xticks = list(cycles_dict.values())[0][region].index[::20]
    plt.xticks(xticks, rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το ενιαίο διάγραμμα των κυκλικών συνιστωσών για {region} ως: {filename}")

def compute_volatilities_dual(cycles_dict):
    """
    Υπολογίζει την τυπική απόκλιση της κυκλικής συνιστώσας για κάθε μεταβλητή, για κάθε περιοχή.
    Επιστρέφει δύο λεξικά: ένα για την Euro και ένα για την Ελλάδα.
    """
    vols_euro = {}
    vols_gr = {}
    for var, cycles in cycles_dict.items():
        vols_euro[var] = np.std(cycles["Euro"])
        vols_gr[var] = np.std(cycles["Ελλάδα"])
    return vols_euro, vols_gr

def relative_volatility_table_dual(vols, region_name):
    """
    Υπολογίζει τη σχετική μεταβλητότητα για κάθε μεταβλητή σε σχέση με την κυκλική συνιστώσα του ΑΕΠ,
    για την περιοχή (region_name: 'Euro' ή 'Ελλάδα').
    Επιστρέφει DataFrame με στήλες: Μεταβλητή, Μεταβλητότητα, Σχετική Μεταβλητότητα.
    """
    base_vol = vols.get("ΑΕΠ", None)
    if base_vol is None or base_vol == 0:
        print(f"Σφάλμα: Η μεταβλητότητα του ΑΕΠ για {region_name} είναι μηδενική ή λείπει.")
        return None
    rel_vol = {var: vol/base_vol for var, vol in vols.items()}
    df_table = pd.DataFrame({
        "Μεταβλητή": list(rel_vol.keys()),
        "Μεταβλητότητα": list(vols.values()),
        "Σχετική Μεταβλητότητα": list(rel_vol.values())
    })
    return df_table

# ----------------------------
# Κύρια Εκτέλεση του Script
# ----------------------------

def main():
    # Λεξικό για την αποθήκευση των δεδομένων (DataFrame με στήλες "Euro" και "Ελλάδα") για κάθε μεταβλητή
    data_dict = {}
    
    for var, sheet in sheet_names_ex6.items():
        print(f"Φόρτωση {var} από το {sheet} στο {excel_file}...")
        try:
            df = load_and_clean_sheet(sheet)
            data_dict[var] = df
            print(f"Το {var} φορτώθηκε με {len(df)} παρατηρήσεις.\n")
        except Exception as e:
            print(f"Σφάλμα κατά τη φόρτωση του {var} από το {sheet}: {e}")
    
    # Λεξικό για αποθήκευση των κυκλικών συνιστωσών για κάθε μεταβλητή και περιοχή
    cycles_all = {}
    
    # Βήμα 1: Σχεδίαση πραγματικής τιμής και τάσης για κάθε μεταβλητή (2 υποπλοτ: Euro και Ελλάδα)
    for var, df in data_dict.items():
        filename = var.replace(" ", "_") + "_Πραγματική_και_Τάση.png"
        plot_actual_vs_trend_dual(df, var, filename)
        
        # Υπολογισμός και αποθήκευση κυκλικών συνιστωσών
        cycle_euro, _ = compute_hp_decomposition(df["Euro"], lamb=1600)
        cycle_gr, _ = compute_hp_decomposition(df["Ελλάδα"], lamb=1600)
        cycles_all[var] = {"Euro": cycle_euro, "Ελλάδα": cycle_gr}
        
        # Σχεδίαση κυκλικής συνιστώσας για κάθε μεταβλητή (2 υποπλοτ)
        filename_cycle = var.replace(" ", "_") + "_Κυκλική.png"
        plot_cyclical_dual(df, var, filename_cycle)
    
    # Βήμα 2: Σχεδίαση συγκριτικών διαγραμμάτων όλων των κυκλικών συνιστωσών για κάθε περιοχή
    plot_all_cyclical_components_dual(cycles_all, "Euro", "all_cyclical_components_Euro.png")
    plot_all_cyclical_components_dual(cycles_all, "Ελλάδα", "all_cyclical_components_Ελλάδα.png")
    
    # Βήμα 3: Υπολογισμός μεταβλητότητας για κάθε μεταβλητή, για κάθε περιοχή
    vols_euro, vols_gr = compute_volatilities_dual(cycles_all)
    for var in vols_euro:
        print(f"Μεταβλητότητα {var} (Euro): {vols_euro[var]:.4f}")
    for var in vols_gr:
        print(f"Μεταβλητότητα {var} (Ελλάδα): {vols_gr[var]:.4f}")
    
    # Βήμα 4: Υπολογισμός σχετικής μεταβλητότητας ως προς το ΑΕΠ για κάθε περιοχή
    rel_vol_euro = relative_volatility_table_dual(vols_euro, "Euro")
    rel_vol_gr = relative_volatility_table_dual(vols_gr, "Ελλάδα")
    
    print("\nΠίνακας Σχετικών Μεταβλητοτήτων για Euro:")
    print(rel_vol_euro)
    if rel_vol_euro is not None:
        rel_vol_euro.to_csv("relative_volatility_Euro.csv", index=False)
        print("Αποθηκεύτηκε ο πίνακας σχετικών μεταβλητοτήτων για Euro ως: relative_volatility_Euro.csv")
    
    print("\nΠίνακας Σχετικών Μεταβλητοτήτων για Ελλάδα:")
    print(rel_vol_gr)
    if rel_vol_gr is not None:
        rel_vol_gr.to_csv("relative_volatility_Ελλάδα.csv", index=False)
        print("Αποθηκεύτηκε ο πίνακας σχετικών μεταβλητοτήτων για Ελλάδα ως: relative_volatility_Ελλάδα.csv")

if __name__ == '__main__':
    main()