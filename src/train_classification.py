from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CLASSIFICATION_TARGET,
    FEATURE_SET_DISCLOSURE,
    FEATURE_SET_OPERATIONAL,
    LEAKAGE_COLS,
    RANDOM_STATE,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
)
from src.evaluation import (
    fit_classification_model,
    illustrative_cost,
    save_confusion_matrix_plot,
    save_feature_importance,
    save_roc_overlay,
)
from src.load_data import load_dataset
from src.preprocess import clean_data, company_train_test_split


def classification_drop_cols() -> list[str]:
    return [*LEAKAGE_COLS, CLASSIFICATION_TARGET]


def _classification_models() -> list[tuple[str, object, bool]]:
    return [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            False,
        ),
        (
            "linear_svc",
            LinearSVC(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced"),
            True,
        ),
        (
            "rbf_svc",
            SVC(
                kernel="rbf",
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            True,
        ),
        (
            "decision_tree_classifier",
            DecisionTreeClassifier(
                max_depth=TREE_MAX_DEPTH,
                min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            False,
        ),
    ]


def run_all_feature_sets() -> dict:
    df = clean_data(load_dataset())
    train_df, test_df = company_train_test_split(df)
    drop = classification_drop_cols()
    all_results: dict = {}

    for feature_set in (FEATURE_SET_OPERATIONAL, FEATURE_SET_DISCLOSURE):
        set_results: dict = {}
        roc_curves: list[tuple[str, object, object]] = []

        for name, est, use_df in _classification_models():
            pipe, metrics, y_test, pred_test, score_test = fit_classification_model(
                train_df,
                test_df,
                est,
                drop,
                feature_set,
                use_decision_function=use_df,
                skip_cv=(name == "rbf_svc"),
            )
            cost = illustrative_cost(y_test, pred_test)
            set_results[name] = {
                "model": name,
                "target": CLASSIFICATION_TARGET,
                "feature_set": feature_set,
                "metrics": metrics,
                "illustrative_cost_test": cost,
            }
            save_confusion_matrix_plot(
                y_test,
                pred_test,
                f"{name} — test ({feature_set})",
                f"classification_cm_{name}_{feature_set}.png",
            )
            if score_test is not None:
                roc_curves.append((name, y_test, score_test))

            if name == "decision_tree_classifier":
                set_results["top_feature_importance"] = save_feature_importance(
                    pipe, feature_set, "decision_tree_classifier"
                )

        save_roc_overlay(roc_curves, feature_set)
        all_results[feature_set] = set_results

    return {
        "task": "classification",
        "feature_sets": all_results,
    }


def train_and_evaluate() -> dict:
    """Legacy entry: operational set, logistic regression only."""
    out = run_all_feature_sets()
    logit = out["feature_sets"][FEATURE_SET_OPERATIONAL]["logistic_regression"]
    logit["model"] = "LogisticRegression"
    return logit
