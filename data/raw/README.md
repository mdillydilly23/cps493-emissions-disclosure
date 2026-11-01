# Raw data

1. Accept the Kaggle license for [Carbon Emission Forecasting Dataset](https://www.kaggle.com/datasets/freshersstaff/carbon-emission-forecasting-dataset).
2. Download **`carbon_emission_dataset_with_Industry.csv`** and save as **`carbon_emission.csv`** in this folder (same contents; shorter local name). The emissions target column is named `Carbon_Emission_tCO2e_TARGET` on Kaggle; `load_dataset()` renames it to `Carbon_Emission_tCO2e`.

If `carbon_emission.csv` is missing, scripts fall back to **`carbon_emission_sample.csv`** (20 rows) for smoke testing.

## Previous dataset (deprecated)

The original [Green Miles](https://www.kaggle.com/datasets/chaitanyajamble/green-milescorporate-logistics-and-carbon-emissions) listing claimed ~2,500 companies but the published download contained only ~25 rows embedded in a `README.md` file. The project switched to the Carbon Emission Forecasting dataset for sufficient sample size.
