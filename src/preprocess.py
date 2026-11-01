from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLASSIFICATION_TARGET,
    COMPANY_COL,
    DATE_COL,
    HIGH_EMITTER_TRAIN_QUANTILE,
    LEAKAGE_COLS,
    RANDOM_STATE,
    REGRESSION_TARGET,
    TEST_COMPANY_FRACTION,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[COMPANY_COL])
    if REGRESSION_TARGET not in out.columns:
        raise ValueError(
            f"Expected column {REGRESSION_TARGET!r}. "
            "Download carbon_emission_dataset_with_Industry.csv from Kaggle "
            "and save as data/raw/carbon_emission.csv"
        )

    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
        out["Year"] = out[DATE_COL].dt.year
        out["Month"] = out[DATE_COL].dt.month

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in (CLASSIFICATION_TARGET,):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    medians = out[numeric_cols].median(numeric_only=True)
    out[numeric_cols] = out[numeric_cols].fillna(medians)

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str).fillna("Unknown")

    return out


def add_high_emitter_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = train_df[REGRESSION_TARGET].quantile(HIGH_EMITTER_TRAIN_QUANTILE)
    train = train_df.copy()
    test = test_df.copy()
    train[CLASSIFICATION_TARGET] = (
        train[REGRESSION_TARGET] > threshold
    ).astype(int)
    test[CLASSIFICATION_TARGET] = (test[REGRESSION_TARGET] > threshold).astype(
        int
    )
    return train, test


def feature_column_lists(
    df: pd.DataFrame, extra_drop: list[str] | None = None
) -> tuple[list[str], list[str]]:
    drop = {
        COMPANY_COL,
        DATE_COL,
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
        *LEAKAGE_COLS,
        *(extra_drop or []),
    }
    candidates = [c for c in df.columns if c not in drop]
    numeric = [
        c
        for c in candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


def company_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    companies = df[COMPANY_COL].unique()
    if len(companies) < 5:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            df, test_size=TEST_COMPANY_FRACTION, random_state=RANDOM_STATE
        )
        return train, test

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_COMPANY_FRACTION, random_state=RANDOM_STATE
    )
    idx_train, idx_test = next(splitter.split(df, groups=df[COMPANY_COL]))
    return df.iloc[idx_train], df.iloc[idx_test]


def make_feature_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features),
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        )
    if not transformers:
        raise ValueError("No feature columns available for modeling.")
    return ColumnTransformer(transformers=transformers)


def build_xy(
    df: pd.DataFrame, target: str, drop_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    always_drop = {COMPANY_COL, DATE_COL, target, *drop_cols, *LEAKAGE_COLS}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    y = df[target].values
    return X, y
