"""
Analysis module.

Three core analyses:
  1. Decomposition fragility — rank instability by CAC source bin.
  2. Signal comparison — Spearman ρ and top-quartile precision per signal.
  3. Sector heterogeneity — precision gap (signal_C − signal_A) per sector.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


# ====================================================================== #
# Analysis 1 — Decomposition Fragility
# ====================================================================== #

def run_decomposition_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin companies by cac_source_pct and measure rank_delta distribution
    + inversion rate per bin.

    Returns:
        (summary_df, df_with_bins) — bin-level stats and the full df
        with cac_bin column attached (needed for plotting).
    """
    bins = [0, 0.25, 0.50, 0.75, 1.01]
    labels = [
        "LTV-Driven (0–25%)",
        "Mixed-Low (25–50%)",
        "Mixed-High (50–75%)",
        "CAC-Driven (75–100%)",
    ]

    df = df.copy()
    df["cac_bin"] = pd.cut(df["cac_source_pct"], bins=bins, labels=labels)

    summary = (
        df.groupby("cac_bin", observed=False)
        .agg(
            mean_rank_delta=("rank_delta", "mean"),
            median_rank_delta=("rank_delta", "median"),
            std_rank_delta=("rank_delta", "std"),
            inversion_rate=("inversion_flag", "mean"),
            n=("rank_delta", "count"),
        )
        .reset_index()
    )

    return summary, df


# ====================================================================== #
# Analysis 2 — Signal Comparison
# ====================================================================== #

def run_signal_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate each signal's predictive power for stressed rank.

    Metrics per (signal, scope):
      - Spearman ρ vs rank_stressed
      - Top-quartile precision: fraction of signal top-25% that
        remain top-25% under stress
    """
    results: list[dict] = []
    signal_cols = ["signal_A", "signal_B", "signal_C"]
    scopes = ["overall"] + sorted(df["sector"].dropna().unique().tolist())

    for signal in signal_cols:
        for scope in scopes:
            subset = df if scope == "overall" else df[df["sector"] == scope]
            valid = subset.dropna(subset=[signal, "rank_stressed"])

            if len(valid) < 10:
                continue

            rho, pval = spearmanr(valid[signal], valid["rank_stressed"])

            top_mask = valid[signal] >= 0.75
            top_count = top_mask.sum()
            if top_count > 0:
                tq_precision = valid.loc[top_mask, "rank_stressed"].ge(0.75).mean()
            else:
                tq_precision = np.nan

            results.append({
                "signal": signal,
                "scope": scope,
                "spearman_rho": rho,
                "p_value": pval,
                "top_quartile_precision": tq_precision,
                "n": len(valid),
            })

    return pd.DataFrame(results)


# ====================================================================== #
# Analysis 3 — Sector Heterogeneity
# ====================================================================== #

def run_sector_analysis(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector, compute precision gap = signal_C precision − signal_A
    precision. Returns a table suitable for heatmap plotting.
    """
    sector_rows = signal_df[signal_df["scope"] != "overall"].copy()

    a_prec = (
        sector_rows[sector_rows["signal"] == "signal_A"][
            ["scope", "top_quartile_precision"]
        ]
        .rename(columns={"top_quartile_precision": "prec_A"})
    )
    c_prec = (
        sector_rows[sector_rows["signal"] == "signal_C"][
            ["scope", "top_quartile_precision"]
        ]
        .rename(columns={"top_quartile_precision": "prec_C"})
    )
    b_prec = (
        sector_rows[sector_rows["signal"] == "signal_B"][
            ["scope", "top_quartile_precision"]
        ]
        .rename(columns={"top_quartile_precision": "prec_B"})
    )

    merged = a_prec.merge(c_prec, on="scope").merge(b_prec, on="scope", how="left")
    merged["precision_gap"] = merged["prec_C"] - merged["prec_A"]

    return merged.sort_values("precision_gap", ascending=False).reset_index(drop=True)
