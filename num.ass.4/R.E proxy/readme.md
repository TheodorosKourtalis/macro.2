US Inflation Proxy  
1-mo ahead CPI via error-weighted Prophet ensemble

Run:  
python expected_inflation_ra.py --start 1990-01-01 --end 2025-04-30 --vintage off --plot on

Outputs:  
expected_inflation.csv  
error_stats.csv  
actual_vs_all_proxies.png  
actual_vs_Rational Exp (IN).png  
Rational_Exp(OOS)_vs_mich.png  
adaptive_vs_Rational_Exp(OOS).png  
forecast_errors.png

Reqs: pandas numpy pandas_datareader prophet bayesian-optimization matplotlib seaborn joblib

Data: FRED/ALFRED series (CPIAUCSL, CPILFESL, UNRATE, FEDFUNDS, DCOILWTICO, M2SL, PPIACO, UMCSENT, MICH)

Code: expected_inflation_ra.py v1.2
