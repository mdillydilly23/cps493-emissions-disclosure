from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import CLASSIFICATION_TARGET, LEAKAGE_COLS, REGRESSION_TARGET, RESULTS_DIR
from src.load_data import load_dataset
from src.preprocess import (
    build_xy,
    clean_data,
    company_train_test_split,
    feature_column_lists,
    make_feature_pipeline,
)


def regression_drop_cols() -> list[str]:
    return [*LEAKAGE_COLS, CLASSIFICATION_TARGET]


def train_and_evaluate() -> dict:
    df = clean_data(load_dataset())
    train_df, test_df = company_train_test_split(df)

    numeric, categorical = feature_column_lists(train_df, regression_drop_cols())
    drop = regression_drop_cols()
    X_train, y_train = build_xy(train_df, REGRESSION_TARGET, drop)
    X_test, y_test = build_xy(test_df, REGRESSION_TARGET, drop)

    model = Pipeline(
        [
            ("prep", make_feature_pipeline(numeric, categorical)),
            ("reg", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }

    pred_test = model.predict(X_test)
    _save_regression_plot(y_test, pred_test)

    out = {
        "task": "regression",
        "target": REGRESSION_TARGET,
        "model": "LinearRegression",
        "metrics": metrics,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_companies": int(train_df["Company_ID"].nunique()),
        "n_test_companies": int(test_df["Company_ID"].nunique()),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "regression_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def _save_regression_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("Actual carbon emissions (tCO₂e)")
    ax.set_ylabel("Predicted")
    ax.set_title("Linear regression — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "regression_test_scatter.png", dpi=150)
    plt.close(fig)
