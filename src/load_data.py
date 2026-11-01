from pathlib import Path

import pandas as pd

from src.config import FULL_DATA, SAMPLE_DATA


def resolve_data_path() -> Path:
    if FULL_DATA.exists():
        return FULL_DATA
    if SAMPLE_DATA.exists():
        return SAMPLE_DATA
    raise FileNotFoundError(
        f"No dataset found. Place Kaggle CSV at {FULL_DATA} or keep {SAMPLE_DATA}."
    )


def load_dataset() -> pd.DataFrame:
    path = resolve_data_path()
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    return df
