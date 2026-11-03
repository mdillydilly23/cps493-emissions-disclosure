from __future__ import annotations

import json
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_TARGET,
    COMPANY_COL,
    COST_FN_WEIGHT,
    COST_FP_WEIGHT,
    CV_FOLDS,
    RANDOM_STATE,
    REGRESSION_TARGET,
    RESULTS_DIR,
)
from src.preprocess import (
    add_high_emitter_labels,
    apply_high_emitter_labels,
    build_xy,
    feature_column_lists,
    high_emitter_threshold,
    make_feature_pipeline,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def illustrative_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = float(COST_FN_WEIGHT * fn + COST_FP_WEIGHT * fp)
    return {
        "false_negatives": float(fn),
        "false_positives": float(fp),
        "true_positives": float(tp),
        "true_negatives": float(tn),
        "illustrative_cost": total,
        "fn_weight": float(COST_FN_WEIGHT),
        "fp_weight": float(COST_FP_WEIGHT),
    }


def grouped_regression_cv(
    train_df: pd.DataFrame,
    estimator,
    drop_cols: list[str],
    feature_set: str,
) -> dict[str, float]:
    numeric, categorical = feature_column_lists(
        train_df, drop_cols, feature_set=feature_set
    )
    prep = make_feature_pipeline(numeric, categorical)
    pipe = Pipeline([("prep", prep), ("model", estimator)])

    X, y = build_xy(train_df, REGRESSION_TARGET, drop_cols, feature_set=feature_set)
    groups = train_df[COMPANY_COL].values
    n_splits = min(CV_FOLDS, len(np.unique(groups)))
    if n_splits < 2:
        return {"rmse_mean": float("nan"), "rmse_std": float("nan")}

    gkf = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        X_tr, y_tr = build_xy(
            fold_train, REGRESSION_TARGET, drop_cols, feature_set=feature_set
        )
        X_va, y_va = build_xy(
            fold_val, REGRESSION_TARGET, drop_cols, feature_set=feature_set
        )
        model = clone(pipe)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_error(y_va, pred)))

    return {"rmse_mean": float(np.mean(scores)), "rmse_std": float(np.std(scores))}


def grouped_classification_cv(
    train_df: pd.DataFrame,
    estimator,
    drop_cols: list[str],
    feature_set: str,
    score_fn: Callable | None = None,
) -> dict[str, float]:
    numeric, categorical = feature_column_lists(
        train_df, drop_cols, feature_set=feature_set
    )
    prep = make_feature_pipeline(numeric, categorical)
    pipe = Pipeline([("prep", prep), ("model", estimator)])

    groups = train_df[COMPANY_COL].values
    n_splits = min(CV_FOLDS, len(np.unique(groups)))
    if n_splits < 2:
        return {"f1_mean": float("nan"), "f1_std": float("nan")}

    gkf = GroupKFold(n_splits=n_splits)
    f1_scores: list[float] = []
    for train_idx, val_idx in gkf.split(train_df, groups=groups):
        fold_train_raw = train_df.iloc[train_idx]
        fold_val_raw = train_df.iloc[val_idx]
        thr = high_emitter_threshold(fold_train_raw[REGRESSION_TARGET])
        fold_train = apply_high_emitter_labels(fold_train_raw, thr)
        fold_val = apply_high_emitter_labels(fold_val_raw, thr)

        X_tr, y_tr = build_xy(
            fold_train, CLASSIFICATION_TARGET, drop_cols, feature_set=feature_set
        )
        X_va, y_va = build_xy(
            fold_val, CLASSIFICATION_TARGET, drop_cols, feature_set=feature_set
        )
        model = clone(pipe)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        f1_scores.append(f1_score(y_va, pred, zero_division=0))

    return {"f1_mean": float(np.mean(f1_scores)), "f1_std": float(np.std(f1_scores))}


def fit_regression_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    estimator,
    drop_cols: list[str],
    feature_set: str,
) -> tuple[Pipeline, dict[str, Any]]:
    numeric, categorical = feature_column_lists(
        train_df, drop_cols, feature_set=feature_set
    )
    prep = make_feature_pipeline(numeric, categorical)
    pipe = Pipeline([("prep", prep), ("model", estimator)])

    X_train, y_train = build_xy(
        train_df, REGRESSION_TARGET, drop_cols, feature_set=feature_set
    )
    X_test, y_test = build_xy(
        test_df, REGRESSION_TARGET, drop_cols, feature_set=feature_set
    )
    pipe.fit(X_train, y_train)

    metrics: dict[str, Any] = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("test", X_test, y_test),
    ]:
        pred = pipe.predict(X)
        metrics[name] = regression_metrics(y, pred)

    metrics["cv"] = grouped_regression_cv(train_df, estimator, drop_cols, feature_set)
    return pipe, metrics


def fit_classification_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    estimator,
    drop_cols: list[str],
    feature_set: str,
    use_decision_function: bool = False,
    skip_cv: bool = False,
) -> tuple[Pipeline, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    train_l, test_l = add_high_emitter_labels(train_df, test_df)
    numeric, categorical = feature_column_lists(
        train_l, drop_cols, feature_set=feature_set
    )
    prep = make_feature_pipeline(numeric, categorical)
    pipe = Pipeline([("prep", prep), ("model", estimator)])

    X_train, y_train = build_xy(
        train_l, CLASSIFICATION_TARGET, drop_cols, feature_set=feature_set
    )
    X_test, y_test = build_xy(
        test_l, CLASSIFICATION_TARGET, drop_cols, feature_set=feature_set
    )
    pipe.fit(X_train, y_train)

    metrics: dict[str, Any] = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = pipe.predict(X)
        score = _positive_scores(pipe, X, use_decision_function)
        metrics[name] = classification_metrics(y, pred, score)

    if skip_cv:
        metrics["cv"] = {
            "f1_mean": None,
            "f1_std": None,
            "skipped": True,
            "reason": "RBF SVM cross-validation omitted for runtime; test metrics reported.",
        }
    else:
        metrics["cv"] = grouped_classification_cv(
            train_df, estimator, drop_cols, feature_set
        )

    pred_test = pipe.predict(X_test)
    score_test = _positive_scores(pipe, X_test, use_decision_function)
    return pipe, metrics, y_test, pred_test, score_test


def _positive_scores(
    pipe: Pipeline, X: pd.DataFrame, use_decision_function: bool
) -> np.ndarray | None:
    model = pipe.named_steps["model"]
    if use_decision_function and hasattr(model, "decision_function"):
        return pipe.decision_function(X)
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return None


def save_correlation_plot(df: pd.DataFrame) -> dict[str, float]:
    numeric = df.select_dtypes(include=[np.number])
    if REGRESSION_TARGET not in numeric.columns:
        return {}
    corr = numeric.corr(numeric_only=True)[REGRESSION_TARGET].sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    corr = corr.drop(REGRESSION_TARGET, errors="ignore")
    top = corr.head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Pearson correlation with Carbon_Emission_tCO2e")
    ax.set_title("Top numeric associations with emissions")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "correlations_with_target.png", dpi=150)
    plt.close(fig)
    return {k: float(v) for k, v in top.items()}


def save_regression_rmse_bars(results: dict[str, dict], feature_set: str) -> None:
    names = []
    rmses = []
    for model_name, data in results.items():
        if model_name.endswith("_unlimited"):
            continue
        names.append(model_name)
        rmses.append(data["metrics"]["test"]["rmse"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, rmses, color="#3498db")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"Regression test RMSE — {feature_set} features")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"regression_rmse_{feature_set}.png", dpi=150)
    plt.close(fig)


def save_regression_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="k", linewidths=0.2, s=18)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("Actual tCO₂e")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def save_overfitting_bars(limited: dict, unlimited: dict, feature_set: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Train R²", "Test R²"]
    limited_vals = [limited["train"]["r2"], limited["test"]["r2"]]
    unlimited_vals = [unlimited["train"]["r2"], unlimited["test"]["r2"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, limited_vals, w, label="Depth-limited tree")
    ax.bar(x + w / 2, unlimited_vals, w, label="Unrestricted tree")
    ax.set_xticks(x, labels)
    ax.set_ylabel("R²")
    ax.set_title(f"Tree overfitting — {feature_set} features")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"tree_overfitting_{feature_set}.png", dpi=150)
    plt.close(fig)


def save_feature_importance(
    pipe: Pipeline, feature_set: str, model_label: str
) -> list[tuple[str, float]]:
    tree = pipe.named_steps["model"]
    if not hasattr(tree, "feature_importances_"):
        return []
    prep = pipe.named_steps["prep"]
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(tree.feature_importances_))]
    imp = tree.feature_importances_
    pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [p[0][:40] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    ax.barh(labels, vals, color="#9b59b6")
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_label} — top features ({feature_set})")
    fig.tight_layout()
    safe = model_label.lower().replace(" ", "_")
    fig.savefig(RESULTS_DIR / f"feature_importance_{safe}_{feature_set}.png", dpi=150)
    plt.close(fig)
    return [(a, float(b)) for a, b in pairs[:10]]


def save_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_xlabel("Predicted high emitter")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def save_roc_overlay(
    curves: list[tuple[str, np.ndarray, np.ndarray]], feature_set: str
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, y_true, y_score in curves:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves (test) — {feature_set}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"classification_roc_{feature_set}.png", dpi=150)
    plt.close(fig)


def write_json(path_name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / path_name).write_text(json.dumps(payload, indent=2))
