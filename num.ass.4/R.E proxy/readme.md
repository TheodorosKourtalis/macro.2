# Expected Inflation Rolling Analysis

This repository provides tools to perform a rolling regression analysis on expected inflation data and to compute error statistics for different window sizes.

## Repository Structure

* **expected\_inflation.csv**: Input dataset containing historical expected inflation values per period.
* **expected\_inflation\_ra.py**: Python script that:

  1. Reads `expected_inflation.csv`.
  2. Applies a rolling regression analysis over a range of window sizes.
  3. Computes error statistics (e.g., MAE, RMSE) for each window.
  4. Writes results to `error_stats.csv`.
* **error\_stats.csv**: Output file with computed error metrics for each window size tested.
* **README.md**: This document explaining the project.

## Requirements

* Python 3.8 or higher
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [statsmodels](https://www.statsmodels.org/) (for regression)
* [matplotlib](https://matplotlib.org/) (optional, for plotting)

You can install the dependencies via pip:

```bash
pip install pandas numpy statsmodels matplotlib
```

## Usage

1. Place your expected inflation data in `expected_inflation.csv`. The expected format is two columns:

   * `date` in `YYYY-MM-DD`
   * `expected_inflation` (numeric value, e.g., percentage or decimal)

2. **Run via command line**:

   ```bash
   python expected_inflation_ra.py
   ```

3. **Run in Spyder IDE**:

   * Open Spyder and navigate to this project folder.
   * Open `expected_inflation_ra.py` in the editor.
   * Ensure the working directory (in the toolbar) is set to the folder containing `expected_inflation.csv`.
   * Press the Run button (▶️) or use F5 to execute the script.

4. After completion, check `error_stats.csv` for the computed metrics:

   * `window`: rolling window size used in regression
   * `mae`: mean absolute error
   * `rmse`: root mean squared error
   * (other metrics as defined in the script)

## Customization

* **Adjust window range**: Open `expected_inflation_ra.py` and modify the list of window sizes to test.
* **Change input/output paths**: Update the file paths at the top of the script as needed.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is released under the MIT License.

## Author

Theodoros Kourtalis
