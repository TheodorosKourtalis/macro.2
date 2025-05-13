US Inflation Proxy

Inverse-error–weighted Prophet ensemble for one-month-ahead US CPI inflation.

Run

python expected_inflation_ra.py \
  --start 1990-01-01 \
  --end   2025-04-30 \
  --vintage off \
  --plot   on

Outputs
	•	expected_inflation.csv
	•	error_stats.csv
	•	actual_vs_all_proxies.png
	•	actual_vs_Rational Exp (IN).png
	•	Rational_Exp(OOS)_vs_mich.png
	•	adaptive_vs_Rational_Exp(OOS).png
	•	forecast_errors.png

Requirements

pandas numpy pandas_datareader prophet bayesian-optimization matplotlib seaborn joblib

Data

All series from FRED/ALFRED. Code in expected_inflation_ra.py (v1.2).
