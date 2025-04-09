#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Άσκηση 6: Υπολογισμός Τάσης και Κυκλικής Συνιστώσας με το φίλτρο HP για Ευρωζώνη και Ελλάδα

Μεταβλητές (σε πραγματικούς όρους):
    - ΑΕΠ (Value added, gross)
    - Ιδιωτική Κατανάλωση (Final consumption expenditure)
    - Επενδύσεις (Gross fixed capital formation)

Για κάθε μεταβλητή διαβάζουμε:
    - Nominal: Τρέχουσες τιμές (Current prices, million euro).
    - Deflator: Implicit deflator (2015=100).

Υπολογίζουμε το Real ως:
    Real = Nominal / (Deflator/100).

Στη συνέχεια:
  1. Εφαρμόζουμε το φίλτρο HP (λ = 1600) για εξαγωγή τάσης και κυκλικής συνιστώσας.
  2. Σχεδιάζουμε σε ένα διάγραμμα (2 υποπλοτ: πάνω Euro, κάτω Ελλάδα) την πραγματική τιμή (Real) και την τάση.
  3. Σχεδιάζουμε σε άλλο διάγραμμα (2 υποπλοτ) την κυκλική συνιστώσα.
  4. Σχεδιάζουμε συγκριτικά διαγράμματα όλων των κυκλικών συνιστωσών (Euro και Ελλάδα).
  5. Υπολογίζουμε τη μεταβλητότητα (τυπική απόκλιση) της κυκλικής συνιστώσας και τις σχετικές μεταβλητότητες (σε σχέση με το ΑΕΠ).
  6. Εξάγουμε πίνακες (DataFrame) με τις σχετικές μεταβλητότητες για Euro και για Ελλάδα.

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

# ----------------------------------------------------------------------------
# 1. Χαρτογράφηση των φύλλων που περιέχουν τις Nominal και Deflator τιμές
#    για τις 3 μεταβλητές που μας ενδιαφέρουν.
# ----------------------------------------------------------------------------
# Προσαρμόστε τα sheet names με βάση το Quarterly_Data.xlsx που έχετε.
# Εδώ ως παράδειγμα:
#   - Value added, gross:        Nominal = Sheet 41,  Deflator = Sheet 80
#   - Final consumption exp.:    Nominal = Sheet 42,  Deflator = Sheet 81
#   - Gross fixed capital form.: Nominal = Sheet 51,  Deflator = Sheet 90
sheet_names_ex6 = {
    "ΑΕΠ": {
        "Nominal": "Sheet 41",    # Current prices, million euro (Value added, gross)
        "Deflator": "Sheet 80"    # Price index (implicit deflator), 2015=100 (Value added, gross)
    },
    "Ιδιωτική Κατανάλωση": {
        "Nominal": "Sheet 42",    # Current prices, million euro (Final consumption expenditure)
        "Deflator": "Sheet 81"    # Price index (implicit deflator), 2015=100 (Final consumption expenditure)
    },
    "Επενδύσεις": {
        "Nominal": "Sheet 51",    # Current prices, million euro (Gross fixed capital formation)
        "Deflator": "Sheet 90"    # Price index (implicit deflator), 2015=100 (Gross fixed capital formation)
    }
}

excel_file = "Quarterly_Data.xlsx"

# ----------------------------------------------------------------------------
# Συναρτήσεις Φόρτωσης & Καθαρισμού Δεδομένων
# ----------------------------------------------------------------------------

def clean_cell(cell):
    """
    Καθαρίζει μια τιμή κελιού:
      - Επιστρέφει np.nan εάν το κελί είναι ":".
      - Ελέγχει για αριθμητική τιμή (float), αγνοώντας τυχόν σύμβολα ή γράμματα.
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

    Προϋποθέσεις:
      - Η σειρά 10 (index 9) περιέχει τις ετικέτες των περιόδων (π.χ., "1995-Q1", "1995-Q2", …).
      - Η σειρά 12 (index 11) περιέχει δεδομένα για Euro area.
      - Η σειρά 13 (index 12) περιέχει δεδομένα για Ελλάδα.

    Επιστρέφει ένα DataFrame με στήλες ["Euro", "Ελλάδα"], με δείκτη τις ετικέτες των περιόδων (quarter_labels).
    """
    df = pd.read_excel(excel_file, sheet_name=sheet, header=None)

    # Λήψη των ετικετών περιόδων από τη σειρά 10 (index 9)
    all_labels = df.iloc[9].values
    # Βρίσκουμε τη θέση του "1995-Q1" (αν υπάρχει)
    try:
        start_idx = list(all_labels).index("1995-Q1")
    except ValueError:
        print(f"Προειδοποίηση: Δεν βρέθηκε το '1995-Q1' στο φύλλο {sheet}. Θα χρησιμοποιηθούν όλες οι διαθέσιμες στήλες.")
        start_idx = 0

    valid_cols = [i for i in range(start_idx, len(all_labels)) if pd.notna(all_labels[i])]
    quarter_labels = [str(all_labels[i]) for i in valid_cols]

    # Διαβάζουμε τα δεδομένα: row 12 (index 11) για Euro, row 13 (index 12) για Ελλάδα
    euro_data = df.iloc[11, valid_cols].apply(clean_cell).values
    gr_data = df.iloc[12, valid_cols].apply(clean_cell).values

    cleaned_df = pd.DataFrame({"Euro": euro_data, "Ελλάδα": gr_data}, index=quarter_labels)
    cleaned_df = cleaned_df.ffill().bfill()
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

# ----------------------------------------------------------------------------
# Συναρτήσεις για το φίλτρο HP & Διαγράμματα
# ----------------------------------------------------------------------------

def compute_hp_decomposition(series, lamb=1600):
    """
    Εφαρμόζει το φίλτρο HP σε μία σειρά (Pandas Series).
    Επιστρέφει:
      - cycle: την κυκλική συνιστώσα
      - trend: την τάση
    """
    cycle, trend = hpfilter(series, lamb=lamb)
    return cycle, trend

def plot_actual_vs_trend_dual(df, var_name, filename):
    """
    Σχεδιάζει σε ένα διάγραμμα (2 υποπλοτ: πάνω Euro, κάτω Ελλάδα) την πραγματική τιμή (df)
    και την τάση που υπολογίζεται με HP φίλτρο.
    """
    cycle_euro, trend_euro = compute_hp_decomposition(df["Euro"], lamb=1600)
    cycle_gr, trend_gr = compute_hp_decomposition(df["Ελλάδα"], lamb=1600)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Euro
    axs[0].plot(df.index, df["Euro"], label="Πραγματική (Euro)", marker='o', linewidth=2)
    axs[0].plot(trend_euro.index, trend_euro, label="Τάση (Euro)", linestyle='--', linewidth=2, color='red')
    axs[0].set_title(f"{var_name} - Ευρωζώνη", fontsize=14)
    axs[0].legend(fontsize=12)

    # Ελλάδα
    axs[1].plot(df.index, df["Ελλάδα"], label="Πραγματική (Ελλάδα)", marker='o', linewidth=2)
    axs[1].plot(trend_gr.index, trend_gr, label="Τάση (Ελλάδα)", linestyle='--', linewidth=2, color='red')
    axs[1].set_title(f"{var_name} - Ελλάδα", fontsize=14)
    axs[1].legend(fontsize=12)

    # Ρυθμίζουμε τα ticks ανά 20 περιόδους
    xticks = df.index[::20]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks, rotation=45)

    fig.suptitle(f"{var_name}: Πραγματική Τιμή και Τάση (HP Filter)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το διάγραμμα (Πραγματική & Τάση) για {var_name} ως: {filename}")

def plot_cyclical_dual(df, var_name, filename):
    """
    Σχεδιάζει την κυκλική συνιστώσα (HP φίλτρο) σε 2 υποπλοτ (πάνω Euro, κάτω Ελλάδα).
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

    fig.suptitle(f"{var_name}: Κυκλική Συνιστώσα (HP Filter)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το διάγραμμα (Κυκλική) για {var_name} ως: {filename}")

def plot_all_cyclical_components_dual(cycles_dict, region, filename):
    """
    Σχεδιάζει όλες τις κυκλικές συνιστώσες (HP filter) για μια περιοχή (Euro ή Ελλάδα) 
    σε ένα ενιαίο διάγραμμα.
    """
    plt.figure(figsize=(14, 7))
    for var, cycles in cycles_dict.items():
        plt.plot(cycles[region].index, cycles[region], marker='o', linewidth=2, label=var)
    plt.title(f"Όλες οι Κυκλικές Συνιστώσες ({region})", fontsize=16)
    plt.xlabel("Περίοδος", fontsize=12)
    plt.ylabel("Κυκλική Συνιστώσα", fontsize=12)
    plt.legend(fontsize=12)

    # Προτείνουμε να παίρνουμε τα xlabels από την πρώτη κυκλική σειρά
    sample_idx = list(cycles_dict.values())[0][region].index
    xticks = sample_idx[::20]
    plt.xticks(xticks, rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Αποθηκεύτηκε το ενιαίο διάγραμμα των κυκλικών συνιστωσών για {region} ως: {filename}")

def compute_volatilities_dual(cycles_dict):
    """
    Υπολογίζει την τυπική απόκλιση της κυκλικής συνιστώσας (cycle) για κάθε μεταβλητή, για κάθε περιοχή.
    Επιστρέφει δύο λεξικά: (vols_euro, vols_gr).
    """
    vols_euro = {}
    vols_gr = {}
    for var, cycles in cycles_dict.items():
        vols_euro[var] = np.std(cycles["Euro"])
        vols_gr[var] = np.std(cycles["Ελλάδα"])
    return vols_euro, vols_gr

def relative_volatility_table_dual(vols, region_name):
    """
    Υπολογίζει τη σχετική μεταβλητότητα κάθε μεταβλητής σε σχέση με το ΑΕΠ (εφόσον υπάρχει στο λεξικό).
    Επιστρέφει ένα DataFrame με: [Μεταβλητή, Μεταβλητότητα, Σχετική Μεταβλητότητα].
    """
    base_vol = vols.get("ΑΕΠ", None)
    if base_vol is None or base_vol == 0:
        print(f"Σφάλμα: Η μεταβλητότητα του ΑΕΠ για {region_name} είναι μηδενική ή δεν υπάρχει.")
        return None
    
    rel_vol = {var: (vol / base_vol) for var, vol in vols.items()}
    df_table = pd.DataFrame({
        "Μεταβλητή": list(rel_vol.keys()),
        "Μεταβλητότητα": list(vols.values()),
        "Σχετική Μεταβλητότητα": list(rel_vol.values())
    })
    return df_table

# ----------------------------------------------------------------------------
# Κύρια Εκτέλεση
# ----------------------------------------------------------------------------

def main():
    # Θα αποθηκεύουμε εδώ τα πραγματικά (Real) DataFrame για κάθε μεταβλητή.
    data_dict = {}

    # 1. Φόρτωση & Υπολογισμός Real για κάθε μεταβλητή (ΑΕΠ, Ιδιωτική Κατανάλωση, Επενδύσεις)
    for var, sheets in sheet_names_ex6.items():
        try:
            print(f"Φόρτωση Nominal για '{var}' από το '{sheets['Nominal']}'...")
            df_nom = load_and_clean_sheet(sheets["Nominal"])
            
            print(f"Φόρτωση Deflator για '{var}' από το '{sheets['Deflator']}'...")
            df_def = load_and_clean_sheet(sheets["Deflator"])

            # Βρίσκουμε το κοινό index (ώστε να υπάρχει αντιστοιχία στις περιόδους)
            common_index = df_nom.index.intersection(df_def.index)
            if common_index.empty:
                print(f"Προειδοποίηση: Δεν υπάρχει κοινό εύρος για '{var}'. Παραλείπεται.")
                continue
            
            df_nom = df_nom.loc[common_index]
            df_def = df_def.loc[common_index]
            
            # Υπολογίζουμε Real = Nominal / (Deflator/100)
            df_real = df_nom / (df_def / 100.0)
            
            # Αποθήκευση των πραγματικών τιμών στο data_dict
            data_dict[var] = df_real
            
            print(f"Υπολογίστηκαν {len(df_real)} πραγματικές τιμές για '{var}'.\n")
        except Exception as e:
            print(f"Σφάλμα κατά την επεξεργασία της μεταβλητής '{var}': {e}")

    # 2. Εφαρμογή φίλτρου HP & σχεδιασμός διαγραμμάτων
    #    cycles_all[var] = {"Euro": cycle_series, "Ελλάδα": cycle_series}
    cycles_all = {}

    for var, df_real in data_dict.items():
        # 2α. Διάγραμμα (πραγματική τιμή & τάση)
        filename_trend = f"{var.replace(' ', '_')}_Real_Trend.png"
        plot_actual_vs_trend_dual(df_real, var, filename_trend)

        # 2β. Διάγραμμα (κυκλική συνιστώσα)
        filename_cycle = f"{var.replace(' ', '_')}_Real_Cycle.png"
        plot_cyclical_dual(df_real, var, filename_cycle)

        # Αποθήκευση της κυκλικής συνιστώσας σε λεξικό
        cycle_euro, _ = compute_hp_decomposition(df_real["Euro"], lamb=1600)
        cycle_gr, _ = compute_hp_decomposition(df_real["Ελλάδα"], lamb=1600)
        cycles_all[var] = {"Euro": cycle_euro, "Ελλάδα": cycle_gr}

    # 3. Συγκεντρωτικά διαγράμματα όλων των κυκλικών συνιστωσών (Euro & Ελλάδα)
    if cycles_all:
        plot_all_cyclical_components_dual(cycles_all, "Euro", "all_cyclical_components_Euro.png")
        plot_all_cyclical_components_dual(cycles_all, "Ελλάδα", "all_cyclical_components_Ελλάδα.png")

    # 4. Υπολογισμός μεταβλητότητας (τυπική απόκλιση) των κυκλικών συνιστωσών
    vols_euro, vols_gr = compute_volatilities_dual(cycles_all)

    # 5. Εκτύπωση μεταβλητοτήτων
    print("=== Μεταβλητότητες (τυπ. απόκλιση) Κυκλικής Συνιστώσας - Euro ===")
    for var in vols_euro:
        print(f"  {var}: {vols_euro[var]:.4f}")

    print("\n=== Μεταβλητότητες (τυπ. απόκλιση) Κυκλικής Συνιστώσας - Ελλάδα ===")
    for var in vols_gr:
        print(f"  {var}: {vols_gr[var]:.4f}")

    # 6. Υπολογισμός σχετικής μεταβλητότητας (σε σχέση με το ΑΕΠ) και αποθήκευση σε CSV
    rel_vol_euro = relative_volatility_table_dual(vols_euro, "Euro")
    if rel_vol_euro is not None:
        print("\nΣχετική Μεταβλητότητα (Euro):")
        print(rel_vol_euro)
        rel_vol_euro.to_csv("relative_volatility_Euro.csv", index=False)
        print("Αποθηκεύτηκε σε: relative_volatility_Euro.csv")

    rel_vol_gr = relative_volatility_table_dual(vols_gr, "Ελλάδα")
    if rel_vol_gr is not None:
        print("\nΣχετική Μεταβλητότητα (Ελλάδα):")
        print(rel_vol_gr)
        rel_vol_gr.to_csv("relative_volatility_Ελλάδα.csv", index=False)
        print("Αποθηκεύτηκε σε: relative_volatility_Ελλάδα.csv")

if __name__ == '__main__':
    main()