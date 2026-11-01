from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    COMPANY_COL,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_COMPANY_FRACTION,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[COMPANY_COL])
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(out[NUMERIC_FEATURES].median())
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("Unknown")
    return out


def company_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    companies = df[COMPANY_COL].unique()
    if len(companies) < 5:
        # Tiny sample: row-level split with warning in metrics.
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


def make_feature_pipeline() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_xy(
    df: pd.DataFrame, target: str, drop_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    always_drop = {COMPANY_COL, target, *drop_cols}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    y = df[target].values
    return X, y
