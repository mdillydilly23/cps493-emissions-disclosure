from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import COMPANY_COL, RANDOM_STATE, REGRESSION_TARGET, RESULTS_DIR
from src.load_data import load_dataset
from src.preprocess import clean_data


def _company_features(df: pd.DataFrame) -> pd.DataFrame:
    hold_out = {
        REGRESSION_TARGET,
        "Carbon_Tax_USD",
        "High_Emitter_Flag",
        COMPANY_COL,
    }
    numeric = df.select_dtypes(include=[np.number]).columns
    use = [c for c in numeric if c not in hold_out]
    agg = df.groupby(COMPANY_COL, as_index=False)[use].mean()
    emissions = df.groupby(COMPANY_COL)[REGRESSION_TARGET].mean().rename("mean_emissions")
    agg = agg.merge(emissions, on=COMPANY_COL)
    return agg


def run_kmeans(max_k: int = 8) -> dict:
    df = clean_data(load_dataset())
    company_df = _company_features(df)
    feature_cols = [
        c for c in company_df.columns if c not in (COMPANY_COL, "mean_emissions")
    ]
    X = company_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias: list[float] = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    chosen_k = 4 if max_k >= 4 else max(k_range)
    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    company_df = company_df.copy()
    company_df["cluster"] = labels

    cluster_summary = []
    for cl in sorted(company_df["cluster"].unique()):
        sub = company_df[company_df["cluster"] == cl]
        cluster_summary.append(
            {
                "cluster": int(cl),
                "n_companies": int(len(sub)),
                "mean_emissions_tCO2e": float(sub["mean_emissions"].mean()),
                "example_companies": sub[COMPANY_COL].head(3).tolist(),
            }
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means on company-average features (emissions held out)")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    out = {
        "n_companies": int(len(company_df)),
        "n_features": len(feature_cols),
        "chosen_k": chosen_k,
        "inertia_by_k": {str(k): inertias[i] for i, k in enumerate(k_range)},
        "cluster_summary": cluster_summary,
    }
    (RESULTS_DIR / "kmeans_summary.json").write_text(json.dumps(out, indent=2))
    return out
