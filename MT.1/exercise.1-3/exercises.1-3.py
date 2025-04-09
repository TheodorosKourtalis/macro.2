#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:25:00 2025

@author: thodoreskourtales
"""

"""
Ανάλυση Δεδομένων ΑΕΠ και Δημιουργία Αναλυτικής Αναφοράς σε LaTeX
--------------------------------------------------------------------

Αυτό το σενάριο εκτελεί τα εξής:
  1. Φορτώνει τα δεδομένα ΑΕΠ από το αρχείο .mat.
  2. Υπολογίζει τον GDP Deflator, τους φυσικούς λογαρίθμους (log) και τους ρυθμούς ανάπτυξης.
  3. Επαληθεύει την ταυτότητα:
         Δ(log(Nominal GDP)) = Δ(log(Real GDP)) + Δ(log(GDP Deflator))
  4. Βρίσκει τη Βάση (Base Year): τη χρονική περίοδο στην οποία ο ονομαστικός ΑΕΠ είναι ίσος (ή όσο πιο κοντά γίνεται)
     στον πραγματικό ΑΕΠ.
  5. Σχεδιάζει γραφήματα των ρυθμών ανάπτυξης για κάθε χώρα σε υπο-διαγράμματα και ένα ενιαίο γράφημα (side by side).
     Στους άξονες του χρόνου εμφανίζονται τα έτη (αρχίζοντας από 1996, δεδομένου ότι οι ρυθμοί ανάπτυξης προκύπτουν ως διαφορές).
  6. Δημιουργεί μια αναφορά σε LaTeX που εξηγεί τη μεθοδολογία και τα αποτελέσματα.

Συντάκτης: Θεόδωρος Κούρταλης
Ημερομηνία: 05/03/2025
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Global variable για αποθήκευση του μέγιστου σφάλματος επαλήθευσης
max_identity_diff = None

def main():
    global max_identity_diff
    # Ορίστε τη διαδρομή προς το αρχείο .mat (βεβαιωθείτε ότι το αρχείο βρίσκεται στη σωστή τοποθεσία)
    file_path = 'GDP_data.mat'
    
    # Φόρτωση των δεδομένων
    data = scipy.io.loadmat(file_path)
    mat_keys = [key for key in data.keys() if not key.startswith('__')]
    print("Βρέθηκαν οι μεταβλητές στο αρχείο .mat:", mat_keys)
    
    # Εξαγωγή των βασικών σειρών δεδομένων
    nominal_gdp = data['nominal_gdp']   # Ονομαστικός ΑΕΠ (28,2)
    real_gdp    = data['real_gdp']        # Πραγματικός ΑΕΠ (28,2)
    gdp_index   = data['gdp_index']       # Δείκτης ΑΕΠ (28,2) (προαιρετικά)
    
    print("Διαστάσεις Ονομαστικού ΑΕΠ:", nominal_gdp.shape)
    print("Διαστάσεις Πραγματικού ΑΕΠ:", real_gdp.shape)
    print("Διαστάσεις Δείκτη ΑΕΠ:", gdp_index.shape)
    
    # Εύρεση της βάσης (base year)
    base_year_ea = find_base_year(nominal_gdp[:, 0], real_gdp[:, 0])
    base_year_gr = find_base_year(nominal_gdp[:, 1], real_gdp[:, 1])
    print("Για την Ευρωζώνη, η βάση βρίσκεται στο index:", base_year_ea)
    print("Για την Ελλάδα, η βάση βρίσκεται στο index:", base_year_gr)
    
    # Υπολογισμός του GDP Deflator
    gdp_deflator = (nominal_gdp / real_gdp) * 100
    
    # Λογαριθμικός μετασχηματισμός
    log_nominal_gdp = np.log(nominal_gdp)
    log_real_gdp    = np.log(real_gdp)
    log_deflator    = np.log(gdp_deflator)
    
    # Υπολογισμός ρυθμών ανάπτυξης (πρώτες διαφορές)
    growth_nominal_gdp = np.diff(log_nominal_gdp, axis=0)
    growth_real_gdp    = np.diff(log_real_gdp, axis=0)
    growth_deflator    = np.diff(log_deflator, axis=0)
    
    # Επαλήθευση ταυτότητας
    identity_diff = growth_nominal_gdp - (growth_real_gdp + growth_deflator)
    max_identity_diff = np.max(np.abs(identity_diff))
    print("Μέγιστη απόλυτη διαφορά (θα πρέπει να είναι κοντά στο 0):", max_identity_diff)
    
    # Σχεδίαση γραφημάτων για κάθε χώρα (υπο-διαγράμματα)
    plot_growth_subplots(growth_nominal_gdp[:, 0], growth_real_gdp[:, 0], growth_deflator[:, 0],
                         country='Euro Area', filename_prefix='subplots_euro_area')
    plot_growth_subplots(growth_nominal_gdp[:, 1], growth_real_gdp[:, 1], growth_deflator[:, 1],
                         country='Greece', filename_prefix='subplots_greece')
    
    # Ενιαίο γράφημα που συγκρίνει τα δεδομένα για τις δύο χώρες (side by side)
    plot_side_by_side(growth_nominal_gdp, growth_real_gdp, growth_deflator, filename='side_by_side_growth.png')
    
    # Δημιουργία αναφοράς σε LaTeX
    generate_latex_report(max_identity_diff, base_year_ea, base_year_gr, image_file='side_by_side_growth.png')

def find_base_year(nominal, real):
    """
    Βρίσκει το index της περιόδου όπου ο ονομαστικός ΑΕΠ είναι ίσος (ή όσο πιο κοντά)
    στον πραγματικό ΑΕΠ.
    """
    index = np.argmin(np.abs(nominal - real))
    return index

def plot_growth_subplots(nom_growth, real_growth, defl_growth, country, filename_prefix='plot'):
    """
    Δημιουργεί ένα γράφημα με 3 υπο-διαγράμματα για μια χώρα:
      1. Ρυθμός ανάπτυξης Ονομαστικού ΑΕΠ.
      2. Ρυθμός ανάπτυξης Πραγματικού ΑΕΠ.
      3. Ρυθμός ανάπτυξης GDP Deflator.
      
    Η χρονική κλίμακα (x-axis) εμφανίζει τα έτη, ξεκινώντας από 1996 (δεδομένου ότι οι ρυθμοί ανάπτυξης
    προκύπτουν ως διαφορές).
    """
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), dpi=150)
    line_width = 2
    marker_size = 6
    
    # Υπολογισμός χρονικής σειράς για τους ρυθμούς (αριθμός στοιχείων = T-1)
    years = np.arange(1996, 1996 + len(nom_growth))
    
    axs[0].plot(years, nom_growth, marker='o', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:blue')
    axs[0].set_title(f'{country} - Nominal GDP Growth (Δ log(Yₜ))', fontsize=12)
    axs[0].set_xlabel('Έτη', fontsize=10)
    axs[0].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[0].tick_params(labelsize=9)
    
    axs[1].plot(years, real_growth, marker='s', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:green')
    axs[1].set_title(f'{country} - Real GDP Growth (Δ log(yₜ))', fontsize=12)
    axs[1].set_xlabel('Έτη', fontsize=10)
    axs[1].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[1].tick_params(labelsize=9)
    
    axs[2].plot(years, defl_growth, marker='^', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:red')
    axs[2].set_title(f'{country} - GDP Deflator Growth (Δ log(Pₜ))', fontsize=12)
    axs[2].set_xlabel('Έτη', fontsize=10)
    axs[2].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[2].tick_params(labelsize=9)
    
    fig.tight_layout()
    image_filename = f'{filename_prefix}.png'
    plt.savefig(image_filename)
    print(f"Το γράφημα υπο-διαγραμμάτων για την {country} αποθηκεύτηκε ως: {image_filename}")
    plt.close()

def plot_side_by_side(nom_growth, real_growth, defl_growth, filename='side_by_side_growth.png'):
    """
    Δημιουργεί ένα ενιαίο γράφημα με 3 σειρές και 2 στήλες:
      - Αριστερή στήλη: δεδομένα για την Ευρωζώνη.
      - Δεξιά στήλη: δεδομένα για την Ελλάδα.
    
    Η x-κλίμακα για κάθε γράφημα εμφανίζει τα έτη, ξεκινώντας από 1996.
    """
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
    
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), dpi=150)
    line_width = 2
    marker_size = 6
    # Υπολογισμός χρονικής σειράς (για 28-1 = 27 σημεία, από 1996 έως 2022)
    years = np.arange(1996, 1996 + nom_growth.shape[0])
    
    # Nominal GDP Growth
    axs[0, 0].plot(years, nom_growth[:, 0], marker='o', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:blue')
    axs[0, 0].set_title('Euro Area - Nominal GDP Growth (Δ log(Yₜ))', fontsize=12)
    axs[0, 0].set_xlabel('Έτη', fontsize=10)
    axs[0, 0].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[0, 0].tick_params(labelsize=9)
    
    axs[0, 1].plot(years, nom_growth[:, 1], marker='o', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:blue')
    axs[0, 1].set_title('Greece - Nominal GDP Growth (Δ log(Yₜ))', fontsize=12)
    axs[0, 1].set_xlabel('Έτη', fontsize=10)
    axs[0, 1].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[0, 1].tick_params(labelsize=9)
    
    # Real GDP Growth
    axs[1, 0].plot(years, real_growth[:, 0], marker='s', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:green')
    axs[1, 0].set_title('Euro Area - Real GDP Growth (Δ log(yₜ))', fontsize=12)
    axs[1, 0].set_xlabel('Έτη', fontsize=10)
    axs[1, 0].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[1, 0].tick_params(labelsize=9)
    
    axs[1, 1].plot(years, real_growth[:, 1], marker='s', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:green')
    axs[1, 1].set_title('Greece - Real GDP Growth (Δ log(yₜ))', fontsize=12)
    axs[1, 1].set_xlabel('Έτη', fontsize=10)
    axs[1, 1].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[1, 1].tick_params(labelsize=9)
    
    # GDP Deflator Growth
    axs[2, 0].plot(years, defl_growth[:, 0], marker='^', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:red')
    axs[2, 0].set_title('Euro Area - GDP Deflator Growth (Δ log(Pₜ))', fontsize=12)
    axs[2, 0].set_xlabel('Έτη', fontsize=10)
    axs[2, 0].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[2, 0].tick_params(labelsize=9)
    
    axs[2, 1].plot(years, defl_growth[:, 1], marker='^', markersize=marker_size, linestyle='-', linewidth=line_width, color='tab:red')
    axs[2, 1].set_title('Greece - GDP Deflator Growth (Δ log(Pₜ))', fontsize=12)
    axs[2, 1].set_xlabel('Έτη', fontsize=10)
    axs[2, 1].set_ylabel('Ρυθμός Ανάπτυξης', fontsize=10)
    axs[2, 1].tick_params(labelsize=9)
    
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Το ενιαίο γράφημα (side by side) αποθηκεύτηκε ως: {filename}")
    plt.close()

def generate_latex_report(max_diff, base_year_ea, base_year_gr, image_file):
    """
    Δημιουργεί ένα αρχείο αναφοράς σε LaTeX που περιγράφει αναλυτικά τη μεθοδολογία και τα αποτελέσματα,
    συμπεριλαμβανομένης της εύρεσης της βάσης (base year) για τον ονομαστικό και τον πραγματικό ΑΕΠ,
    καθώς και την παρουσίαση των γραφημάτων.
    """
    latex_content = r"""
\documentclass{article}

\usepackage{fontspec}
\usepackage{polyglossia}
\setdefaultlanguage{greek}
\setotherlanguage{english}
% Ορισμός βασικού font ως Noto Serif (βεβαιωθείτε ότι είναι εγκατεστημένο)
\setmainfont{Noto Serif}
% Ορισμός monospace font για την ορθή απόδοση ελληνικών χαρακτήρων στα \texttt{}
\newfontfamily\greekfonttt{Noto Sans Mono}

\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\title{Αναλυτική Αναφορά Ανάλυσης Δεδομένων ΑΕΠ}
\author{Θεόδωρος Κούρταλης}
\date{\today}
\begin{document}
\maketitle

\section{Εισαγωγή}
Η παρούσα αναφορά παρουσιάζει τη διαδικασία ανάλυσης των δεδομένων ΑΕΠ από το αρχείο \texttt{GDP\_data.mat}. Οι κύριες μεταβλητές είναι ο Ονομαστικός ΑΕΠ, ο Πραγματικός ΑΕΠ και ο Δείκτης ΑΕΠ. Από τον ονομαστικό και τον πραγματικό ΑΕΠ υπολογίστηκε ο GDP Deflator μέσω του τύπου:
\[
\text{GDP Deflator} = \frac{\text{Nominal GDP}}{\text{Real GDP}} \times 100.
\]
Εφαρμόστηκε ο φυσικός λογαρίθμος για τον υπολογισμό των ρυθμών ανάπτυξης (ως πρώτες διαφορές των λογαρίθμων) και επαληθεύτηκε η σχέση:
\[
\Delta \log(\text{Nominal GDP}) = \Delta \log(\text{Real GDP}) + \Delta \log(\text{GDP Deflator}),
\]
με μέγιστη απόλυτη διαφορά: \textbf{""" + f"{max_diff:.5f}" + r"""}. 

\section{Μεθοδολογία}
\begin{enumerate}
    \item \textbf{Φόρτωση Δεδομένων:} Τα δεδομένα εξήχθησαν από το αρχείο \texttt{GDP\_data.mat} χρησιμοποιώντας το \texttt{scipy.io.loadmat}. Οι βασικές σειρές είναι:
    \begin{itemize}
       \item \texttt{nominal\_gdp} --- Ονομαστικός ΑΕΠ.
       \item \texttt{real\_gdp} --- Πραγματικός ΑΕΠ.
       \item \texttt{gdp\_index} --- Δείκτης ΑΕΠ.
    \end{itemize}
    \item \textbf{Υπολογισμός GDP Deflator:} Ο GDP Deflator υπολογίστηκε ως:
    \[
    \text{GDP Deflator} = \frac{\text{Nominal GDP}}{\text{Real GDP}} \times 100.
    \]
    \item \textbf{Λογαριθμικός Μετασχηματισμός και Υπολογισμός Ρυθμών Ανάπτυξης:} Εφαρμόστηκαν οι φυσικοί λογαρίθμοι στις σειρές και στη συνέχεια υπολογίστηκαν οι ρυθμοί ανάπτυξης ως οι πρώτες διαφορές των λογαρίθμων.
    \item \textbf{Εύρεση Βάσης (Base Year):} Η βάση ορίζεται ως η χρονική περίοδος κατά την οποία ο ονομαστικός ΑΕΠ είναι ίσος (ή όσο το δυνατόν πιο κοντά) στον πραγματικό ΑΕΠ. Για την Ευρωζώνη η βάση βρίσκεται στο index \textbf{""" + str(base_year_ea) + r"""}, και για την Ελλάδα στο index \textbf{""" + str(base_year_gr) + r"""}.
    
    \textit{Παρατήρηση:} Και οι σειρές του ονομαστικού και του πραγματικού ΑΕΠ παρουσιάζουν τάση (κλίση), οπότε η εύρεση της βάσης προσφέρει ένα σημείο αναφοράς για τη σύγκριση των μεταβολών.
    
    \item \textbf{Σχεδίαση Γραφημάτων:} Για κάθε χώρα δημιουργήθηκαν:
    \begin{itemize}
        \item Γράφημα υπο-διαγραμμάτων (3 υπογράμματα) που απεικονίζουν τους ρυθμούς ανάπτυξης.
        \item Ένα ενιαίο γράφημα (side by side) που συγκρίνει τους ρυθμούς ανάπτυξης μεταξύ της Ευρωζώνης και της Ελλάδας.
    \end{itemize}
\end{enumerate}

\section{Αποτελέσματα}
Το ενιαίο γράφημα που συγκρίνει τους ρυθμούς ανάπτυξης για τις δύο χώρες παρουσιάζεται στην εικόνα:
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.9\textwidth]{""" + image_file + r"""}
    \caption{Σύνοψη ρυθμών ανάπτυξης για την Ευρωζώνη και για την Ελλάδα. Τα δεδομένα αναπαριστούν τα έτη από 1996 έως 2022.}
\end{figure}

Τα γραφήματα υπο-διαγραμμάτων για κάθε χώρα αποθηκεύτηκαν στα αρχεία:
\begin{itemize}
    \item \texttt{subplots\_euro\_area.png} για την Ευρωζώνη.
    \item \texttt{subplots\_greece.png} για την Ελλάδα.
\end{itemize}


\end{document}
    """
    report_filename = "report.tex"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Η αναφορά σε LaTeX δημιουργήθηκε και αποθηκεύτηκε στο αρχείο: {report_filename}")

if __name__ == '__main__':
    main()