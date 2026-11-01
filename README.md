# Corporate Carbon Emissions — ML Project

**CPS493-01 / CPS536-01 · Machine Learning · SUNY New Paltz**

Predict daily **carbon emissions (tCO₂e)** from operational and ESG-style features (regression), and flag **high-emitter** days (classification). Motivation: corporate carbon forecasting and disclosure accountability.

**Author:** Mitchell Davis

---

## Dataset

[Carbon Emission Forecasting Dataset](https://www.kaggle.com/datasets/freshersstaff/carbon-emission-forecasting-dataset) (Kaggle).

| | Full file (local) | Sample (repo) |
|--|-------------------|---------------|
| Path | `data/raw/carbon_emission.csv` | `data/raw/carbon_emission_sample.csv` |
| Scale | ~18,250 daily rows, 50 companies (2024) | 20 rows (smoke tests) |

Download steps and column naming (`Carbon_Emission_tCO2e_TARGET` → `Carbon_Emission_tCO2e`) are in [`data/raw/README.md`](data/raw/README.md). The full CSV is gitignored (Kaggle license); scripts use the sample only when the full file is missing.

---

## Quick start

### Environment

**Option A — venv (documented default)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — Conda** (e.g. course env `CPS493`): activate it, then `pip install -r requirements.txt` if anything is missing.

Verify imports:

```bash
python scripts/installcheck.py
```

### Run preliminary models

From the repo root:

```bash
python scripts/run_preliminary.py
```

The script prints which data file it uses, then trains both tasks and writes outputs to `results/`.

---

## Machine learning tasks

| Task | Target | Model (preliminary) |
|------|--------|---------------------|
| **A — Regression** | `Carbon_Emission_tCO2e` | Linear regression |
| **B — Classification** | `High_Emitter_Flag` | Logistic regression |

**High-emitter label:** `1` when emissions exceed the **75th percentile on the training split only** (threshold not fit on test data).

**Evaluation split:** `GroupShuffleSplit` on `Company_ID` (~80% / ~20% companies) so test days are from held-out companies, not random rows from the same firm.

**Preprocessing:** parse dates → `Year` / `Month`; coerce numerics; median imputation; one-hot categoricals; `StandardScaler` on numeric features.

---

## Leakage rules

Do **not** use `Carbon_Emission_tCO2e` or `High_Emitter_Flag` as input features. The classification label is derived from emissions; it must never appear in the feature matrix for either task.

---

## Outputs (`results/`)

After `run_preliminary.py`:

| File | Description |
|------|-------------|
| `regression_metrics.json` | Train/test RMSE, MAE, R² |
| `regression_test_scatter.png` | Predicted vs actual (test) |
| `classification_metrics.json` | Train/test accuracy, F1, ROC-AUC |
| `classification_confusion_matrix.png` | Test confusion matrix |
| `classification_roc.png` | Test ROC curve |

---

## Repository layout

| Path | Role |
|------|------|
| `src/config.py` | Paths, targets, random seed, split fraction |
| `src/load_data.py` | Resolve sample vs full CSV; normalize Kaggle column name |
| `src/preprocess.py` | Cleaning, features, company-level split |
| `src/train_regression.py` | Task A |
| `src/train_classification.py` | Task B |
| `scripts/run_preliminary.py` | End-to-end preliminary run |
| `scripts/installcheck.py` | Dependency smoke test |

---

## Roadmap (final project)

| Status | Algorithms & evaluation |
|--------|-------------------------|
| **Done (Oct checkpoint)** | Data pipeline; linear + logistic regression; train/test metrics; ROC (classification) |
| **Planned (Dec)** | SVM and/or decision trees; k-fold cross-validation; model comparison in report |

---

## Note on dataset choice

An early draft used [Green Miles](https://www.kaggle.com/datasets/chaitanyajamble/green-milescorporate-logistics-and-carbon-emissions). The listing describes thousands of companies, but the published download was only **~25 rows** embedded in a `README.md`. This project uses Carbon Emission Forecasting instead for enough data to support company-level train/test evaluation.
