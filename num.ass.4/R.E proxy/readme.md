US Inflation Proxy
==================

A one-month-ahead CPI inflation proxy using an inverse-error–weighted Prophet ensemble.

Run
---

python expected_inflation_ra.py 
–start 1990-01-01 
–end   2025-04-30 
–vintage off 
–plot   on

Outputs
-------
- **expected_inflation.csv**   
  Monthly proxy series + raw model forecasts  
- **error_stats.csv**  
  MAPE, RMSE, Bias for four methods  
- **Plots**  
  - actual_vs_all_proxies.png  
  - actual_vs_Rational Exp (IN).png  
  - Rational_Exp (OOS)_vs_mich.png  
  - adaptive_vs_Rational_Exp (OOS).png  
  - forecast_errors.png  

Requirements
------------
- pandas  
- numpy  
- pandas_datareader  
- prophet  
- bayesian-optimization  
- matplotlib  
- seaborn  
- joblib  

Data
----
All series from FRED/ALFRED:  
CPIAUCSL, CPILFESL, UNRATE, FEDFUNDS, DCOILWTICO, M2SL, PPIACO, UMCSENT, MICH  

Code
----
`expected_inflation_ra.py` v1.2  

Feel free to fork and adapt!
