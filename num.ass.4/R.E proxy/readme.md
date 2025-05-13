# US Inflation Expectations Proxy

A simple, transparent one-month-ahead CPI inflation proxy built as an inverse-error–weighted ensemble of Facebook Prophet models.

## Usage

```bash
git clone https://github.com/TheodorosKourtalis/your-repo.git
cd your-repo
pip install pandas numpy pandas_datareader prophet bayesian-optimization matplotlib seaborn joblib
python expected_inflation_ra.py --start 1990-01-01 --end 2025-04-30 --vintage off --plot on

Outputs
	•	expected_inflation.csv — monthly proxy series and raw forecasts
	•	error_stats.csv      — MAPE, RMSE, Bias for four methods
	•	PNG plots:
	•	actual_vs_all_proxies.png
	•	actual_vs_Rational Exp (IN).png
	•	Rational_Exp(OOS)_vs_mich.png
	•	adaptive_vs_Rational_Exp(OOS).png
	•	forecast_errors.png

Data & Code

All data come from FRED/ALFRED; code is in expected_inflation_ra.py (v1.2).
Feel free to fork and adapt.

