"""
Signal construction module.

Builds three ranking signals for predictive comparison:
  A — static LTV/CAC ratio snapshot (baseline)
  B — ratio_trend velocity only
  C — 50/50 weighted composite of A and B (percentile-normalised)
"""

from __future__ import annotations

import pandas as pd


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct and append signal columns to the dataset.

    Each signal is a within-sector percentile rank (0–1).

    Returns:
        DataFrame with signal_A, signal_B, signal_C columns added.
    """
    df = df.copy()

    df["pct_ratio"] = df.groupby("sector")["ltv_cac_ratio"].rank(pct=True)

    # ratio_trend may contain NaNs from rolling window warm-up;
    # fill with 0 (neutral trend) before ranking
    trend_filled = df["ratio_trend"].fillna(0)
    df["pct_trend"] = trend_filled.groupby(df["sector"]).rank(pct=True)

    # Signals
    df["signal_A"] = df["pct_ratio"]                                # static snapshot
    df["signal_B"] = df["pct_trend"]                                # velocity only
    df["signal_C"] = 0.5 * df["pct_ratio"] + 0.5 * df["pct_trend"] # composite

    return df
