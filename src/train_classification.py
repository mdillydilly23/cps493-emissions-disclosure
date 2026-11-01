from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import CLASSIFICATION_TARGET, LEAKAGE_COLS, RESULTS_DIR
from src.preprocess import build_xy, clean_data, company_train_test_split, make_feature_pipeline
from src.load_data import load_green_miles


def classification_drop_cols() -> list[str]:
    # Greenwashing flag is derived from discrepancy; never use discrepancy or raw both emissions.
    return LEAKAGE_COLS + [CLASSIFICATION_TARGET]


def train_and_evaluate() -> dict:
    df = clean_data(load_green_miles())
    train_df, test_df = company_train_test_split(df)

    drop = classification_drop_cols()
    X_train, y_train = build_xy(train_df, CLASSIFICATION_TARGET, drop)
    X_test, y_test = build_xy(test_df, CLASSIFICATION_TARGET, drop)

    model = Pipeline(
        [
            ("prep", make_feature_pipeline()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        metrics[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        }

    pred_test = model.predict(X_test)
    proba_test = model.predict_proba(X_test)[:, 1]
    _save_confusion_matrix(y_test, pred_test)
    if len(np.unique(y_test)) > 1:
        _save_roc_curve(y_test, proba_test)

    out = {
        "task": "classification",
        "target": CLASSIFICATION_TARGET,
        "model": "LogisticRegression",
        "metrics": metrics,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_xlabel("Predicted greenwashing flag")
    ax.set_ylabel("Actual")
    ax.set_title("Logistic regression — test confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classification_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="Logistic regression")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classification_roc.png", dpi=150)
    plt.close(fig)
