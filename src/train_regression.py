from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CLASSIFICATION_TARGET,
    FEATURE_SET_DISCLOSURE,
    FEATURE_SET_OPERATIONAL,
    LEAKAGE_COLS,
    RANDOM_STATE,
    REGRESSION_TARGET,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
)
from src.evaluation import (
    fit_regression_model,
    save_feature_importance,
    save_overfitting_bars,
    save_regression_rmse_bars,
    save_regression_scatter,
)
from src.load_data import load_dataset
from src.preprocess import build_xy, clean_data, company_train_test_split


def regression_drop_cols() -> list[str]:
    return [*LEAKAGE_COLS, CLASSIFICATION_TARGET]


def _regression_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree_limited": DecisionTreeRegressor(
            max_depth=TREE_MAX_DEPTH,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
        ),
        "decision_tree_unlimited": DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
    }


def run_all_feature_sets() -> dict:
    df = clean_data(load_dataset())
    train_df, test_df = company_train_test_split(df)
    drop = regression_drop_cols()
    all_results: dict = {}

    for feature_set in (FEATURE_SET_OPERATIONAL, FEATURE_SET_DISCLOSURE):
        set_results: dict = {}
        pipes: dict = {}

        for name, est in _regression_models().items():
            pipe, metrics = fit_regression_model(
                train_df, test_df, est, drop, feature_set
            )
            pipes[name] = pipe
            set_results[name] = {
                "model": name,
                "target": REGRESSION_TARGET,
                "feature_set": feature_set,
                "metrics": metrics,
            }

        save_regression_rmse_bars(set_results, feature_set)

        X_test, y_test = build_xy(test_df, REGRESSION_TARGET, drop, feature_set=feature_set)
        save_regression_scatter(
            y_test,
            pipes["linear_regression"].predict(X_test),
            f"Linear regression — test ({feature_set})",
            f"regression_scatter_linear_{feature_set}.png",
        )
        save_regression_scatter(
            y_test,
            pipes["decision_tree_limited"].predict(X_test),
            f"Depth-limited tree — test ({feature_set})",
            f"regression_scatter_tree_{feature_set}.png",
        )

        save_overfitting_bars(
            set_results["decision_tree_limited"]["metrics"],
            set_results["decision_tree_unlimited"]["metrics"],
            feature_set,
        )

        importance = save_feature_importance(
            pipes["decision_tree_limited"],
            feature_set,
            "decision_tree_limited",
        )
        set_results["top_feature_importance"] = importance
        all_results[feature_set] = set_results

    return {
        "task": "regression",
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_companies": int(train_df["Company_ID"].nunique()),
        "n_test_companies": int(test_df["Company_ID"].nunique()),
        "feature_sets": all_results,
    }


def train_and_evaluate() -> dict:
    """Legacy entry: operational set, linear regression only."""
    out = run_all_feature_sets()
    linear = out["feature_sets"][FEATURE_SET_OPERATIONAL]["linear_regression"]
    linear["model"] = "LinearRegression"
    return linear
