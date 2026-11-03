#!/usr/bin/env python3
"""Run full model comparison, clustering, and write results for report/brief."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster_companies import run_kmeans
from src.config import RESULTS_DIR
from src.evaluation import save_correlation_plot, write_json
from src.load_data import load_dataset, resolve_data_path
from src.preprocess import clean_data
from src.train_classification import run_all_feature_sets as run_clf
from src.train_regression import run_all_feature_sets as run_reg


def main() -> None:
    print(f"Using data: {resolve_data_path()}")
    df = clean_data(load_dataset())
    correlations = save_correlation_plot(df)
    print("Saved correlation plot; top |r| features:", list(correlations.keys())[:5])

    reg = run_reg()
    print("Regression complete (operational linear test R²):", end=" ")
    print(reg["feature_sets"]["operational"]["linear_regression"]["metrics"]["test"]["r2"])

    clf = run_clf()
    print("Classification complete (operational logistic test F1):", end=" ")
    print(
        clf["feature_sets"]["operational"]["logistic_regression"]["metrics"]["test"]["f1"]
    )

    kmeans = run_kmeans()
    print(f"K-means: {kmeans['n_companies']} companies, k={kmeans['chosen_k']}")

    payload = {
        "data_path": str(resolve_data_path()),
        "correlations_top": correlations,
        "regression": reg,
        "classification": clf,
        "kmeans": kmeans,
        "notes": {
            "feature_sets": {
                "operational": "All usable predictors including energy and carbon tax.",
                "disclosure": "Energy and carbon tax columns removed for a harder task.",
            },
            "validation": "GroupKFold on training companies; 20% company holdout for test.",
            "cost": "Illustrative cost = 5*FN + 1*FP on test confusion matrices.",
        },
    }
    write_json("model_comparison.json", payload)

    # Backward-compatible single-model JSON for LaTeX report figures
    lin = reg["feature_sets"]["operational"]["linear_regression"]
    write_json(
        "regression_metrics.json",
        {
            "task": "regression",
            "target": lin["target"],
            "model": "LinearRegression",
            "metrics": lin["metrics"],
            "n_train_rows": reg["n_train_rows"],
            "n_test_rows": reg["n_test_rows"],
            "n_train_companies": reg["n_train_companies"],
            "n_test_companies": reg["n_test_companies"],
        },
    )
    logit = clf["feature_sets"]["operational"]["logistic_regression"]
    write_json(
        "classification_metrics.json",
        {
            "task": "classification",
            "target": logit["target"],
            "model": "LogisticRegression",
            "metrics": logit["metrics"],
            "n_train_rows": reg["n_train_rows"],
            "n_test_rows": reg["n_test_rows"],
        },
    )

    _legacy_report_figures()

    print("Wrote results/ (model_comparison.json, figures).")
    print("Optional: python report/build_project_brief.py")


def _legacy_report_figures() -> None:
    """Names expected by report/corporate_carbon_emissions_report.tex."""
    copies = [
        ("regression_scatter_linear_operational.png", "regression_test_scatter.png"),
        (
            "classification_cm_logistic_regression_operational.png",
            "classification_confusion_matrix.png",
        ),
        ("classification_roc_operational.png", "classification_roc.png"),
    ]
    for src, dst in copies:
        s, d = RESULTS_DIR / src, RESULTS_DIR / dst
        if s.exists():
            shutil.copy(s, d)


if __name__ == "__main__":
    main()
