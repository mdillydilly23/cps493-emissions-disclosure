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

| Task | Target | Models |
|------|--------|--------|
| **A — Regression** | `Carbon_Emission_tCO2e` | Linear regression; depth-limited and unrestricted decision trees |
| **B — Classification** | `High_Emitter_Flag` | Logistic regression; linear and RBF SVM; decision tree classifier |

Each task is evaluated on two feature sets: **operational** (all usable predictors) and **disclosure** (energy and carbon-tax columns removed). **GroupKFold** (k=5) on training companies supplements the company holdout test split. A **k-means** check clusters company averages with emissions held out.

**High-emitter label:** `1` when emissions exceed the **75th percentile on the training split only** (threshold not fit on test data).

**Evaluation split:** `GroupShuffleSplit` on `Company_ID` (~80% / ~20% companies) so test days are from held-out companies, not random rows from the same firm.

**Preprocessing:** parse dates → `Year` / `Month`; coerce numerics; `SimpleImputer` + `StandardScaler` on numerics and one-hot categoricals (fit on training data only inside each model pipeline).

---

## Leakage rules

Do **not** use `Carbon_Emission_tCO2e` or `High_Emitter_Flag` as input features. The classification label is derived from emissions; it must never appear in the feature matrix for either task.

---

## Outputs (`results/`)

After `run_preliminary.py`:

| File | Description |
|------|-------------|
| `model_comparison.json` | Full comparison (both feature sets, all models, k-means) |
| `regression_metrics.json` | Operational linear regression (legacy name for report) |
| `classification_metrics.json` | Operational logistic regression (legacy name for report) |
| `regression_test_scatter.png` | Operational linear regression scatter (test) |
| `classification_confusion_matrix.png` | Operational logistic confusion matrix |
| `classification_roc.png` | Operational ROC overlay |
| `correlations_with_target.png`, `regression_rmse_*.png`, `tree_overfitting_*.png`, `classification_roc_*.png`, `kmeans_elbow.png` | Presentation / brief figures |

---

## Repository layout

| Path | Role |
|------|------|
| `src/config.py` | Paths, targets, random seed, split fraction |
| `src/load_data.py` | Resolve sample vs full CSV; normalize Kaggle column name |
| `src/preprocess.py` | Cleaning, features, company-level split |
| `src/train_regression.py` | Task A |
| `src/train_classification.py` | Task B |
| `scripts/run_preliminary.py` | Full model comparison + clustering |
| `report/build_project_brief.py` | Local Word brief (gitignored `report/` folder) |
| `scripts/installcheck.py` | Dependency smoke test |

---

## Roadmap (final project)

| Status | Algorithms & evaluation |
|--------|-------------------------|
| **Done** | Pipeline imputation inside sklearn; operational vs disclosure features; linear + tree regression; logistic + linear/RBF SVM + tree classification; grouped k-fold; k-means; `model_comparison.json` |
| **Presentation** | Local `report/project_brief.docx` via `python report/build_project_brief.py` after a results run |

---

## Note on dataset choice

An early draft used [Green Miles](https://www.kaggle.com/datasets/chaitanyajamble/green-milescorporate-logistics-and-carbon-emissions). The listing describes thousands of companies, but the published download was only **~25 rows** embedded in a `README.md`. This project uses Carbon Emission Forecasting instead for enough data to support company-level train/test evaluation.
