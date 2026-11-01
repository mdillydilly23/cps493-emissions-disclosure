# CPS493 / CPS536 — Corporate Carbon Emissions (ML Project)

Regression and classification on **operational and ESG-style features** to predict daily **carbon emissions (tCO₂e)** and flag **high-emitter** days for companies — motivated by corporate carbon footprint forecasting and disclosure accountability.

**Dataset:** [Carbon Emission Forecasting Dataset](https://www.kaggle.com/datasets/freshersstaff/carbon-emission-forecasting-dataset) (Kaggle).

## Setup (Mac, venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle CSV and place it at:

```text
data/raw/carbon_emission.csv
```

See `data/raw/README.md` for details. A small **sample** file is included so scripts run before you download the full dataset.

## Run preliminary pipeline

```bash
python scripts/run_preliminary.py
```

Outputs metrics under `results/` and figures for your report.

## Project layout

| Path | Purpose |
|------|---------|
| `src/config.py` | Paths, target columns, leakage-safe feature lists |
| `src/load_data.py` | Load CSV |
| `src/preprocess.py` | Cleaning, encoding, company-level train/test split |
| `src/train_regression.py` | Linear regression (Task A) |
| `src/train_classification.py` | Logistic regression (Task B) |
| `scripts/run_preliminary.py` | End-to-end preliminary run |

## Tasks

- **Regression (Task A):** predict `Carbon_Emission_tCO2e` from energy, production, logistics, and related features.
- **Classification (Task B):** predict `High_Emitter_Flag` (emissions above the 75th percentile on the **training** split).

## Course algorithms (planned)

- **Done / preliminary:** Linear regression, logistic regression  
- **Final (Dec):** + SVM and/or decision trees; k-fold CV; ROC for classification  

## Leakage rules

Do not use `Carbon_Emission_tCO2e` or `High_Emitter_Flag` as features. The high-emitter label is derived from emissions using a threshold fit on training data only.

## Dataset change (Fall 2026)

This repo initially used [Green Miles](https://www.kaggle.com/datasets/chaitanyajamble/green-milescorporate-logistics-and-carbon-emissions). That Kaggle page describes ~2,500 companies, but the only downloadable file was a small `README.md` containing **25 rows** of CSV—not the advertised scale. The project moved to the Carbon Emission Forecasting dataset (~3.4 MB CSV, daily company records) for meaningful train/test evaluation.

## Author

Mitchell Davis — Undergraduate project, SUNY New Paltz ML & CPS493-01 / CPS536-01.
