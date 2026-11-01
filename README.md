# CPS493 / CPS536 — Emissions Disclosure Gap (ML Project)

Regression and classification on corporate logistics and emissions data, motivated by gaps between **reported** and **physics-based (calculated)** CO₂ estimates — related to satellite-vs-disclosure validation work (e.g. Thermal Ledger).

**Dataset:** [Green Miles: Corporate Logistics & Carbon Emissions](https://www.kaggle.com/datasets/chaitanyajamble/green-milescorporate-logistics-and-carbon-emissions) (Kaggle).

## Setup (Mac, venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle CSV and place it at:

```text
data/raw/green_miles.csv
```

See `data/raw/README.md` for column expectations. A small **sample** file is included so scripts run before you download the full dataset.

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

## Course algorithms (planned)

- **Done / preliminary:** Linear regression, logistic regression  
- **Final (Dec):** + SVM and/or decision trees; k-fold CV; ROC for classification  

## Leakage rules

Do not use `Reported_CO2_Emissions`, `Calculated_CO2_Emissions`, or `Emissions_Discrepancy` as features when predicting `Greenwashing_Flag`. For regression on `Emissions_Discrepancy`, exclude both emission columns from features.

## Author

Mitchell Davis — Undergraduate project, SUNY New Paltz ML & CPS493-01 / CPS536-01.
