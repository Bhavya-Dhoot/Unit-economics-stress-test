"""
Feature engineering module.

Constructs LTV/CAC proxies, trend features, and CAC source decomposition
from raw FMP quarterly income-statement and growth data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TREND_WINDOW


def build_company_features(
    symbol: str,
    sector: str,
    income_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    ttm: dict,
    profile: dict,
) -> pd.DataFrame:
    """
    Per-company feature construction from raw API data.
    Returns one row per quarterly period with all derived metrics.
    """
    if income_df.empty:
        return pd.DataFrame()

    df = income_df.copy()
    df["sector"] = sector
    df["mktCap"] = profile.get("mktCap", np.nan)

    # ------------------------------------------------------------------ #
    # Revenue and margin primitives
    # ------------------------------------------------------------------ #
    df["revenue"] = pd.to_numeric(df.get("revenue"), errors="coerce")
    df["grossProfit"] = pd.to_numeric(df.get("grossProfit"), errors="coerce")
    df["gross_margin"] = df["grossProfit"] / df["revenue"].replace(0, np.nan)
    df["delta_revenue"] = df["revenue"].diff().clip(lower=1)

    # ------------------------------------------------------------------ #
    # CAC proxy
    # Prefer explicit sellingAndMarketingExpenses;
    # fall back to 40 % of SG&A
    # ------------------------------------------------------------------ #
    sm_col = "sellingAndMarketingExpenses"
    sga_col = "sellingGeneralAndAdministrativeExpenses"

    df[sm_col] = pd.to_numeric(df.get(sm_col), errors="coerce")
    df[sga_col] = pd.to_numeric(df.get(sga_col), errors="coerce")

    df["s_and_m"] = df[sm_col].copy()
    mask_missing_sm = df["s_and_m"].isna() | (df["s_and_m"] == 0)
    df.loc[mask_missing_sm, "s_and_m"] = df.loc[mask_missing_sm, sga_col] * 0.4

    df["cac_proxy"] = df["s_and_m"] / df["delta_revenue"]
    df["cac_proxy"] = df["cac_proxy"].clip(lower=0.01, upper=50)

    # ------------------------------------------------------------------ #
    # Implied churn and LTV proxy
    # ------------------------------------------------------------------ #
    df["revenue_retention"] = df["revenue"] / df["revenue"].shift(1)
    df["implied_churn"] = (1 - df["revenue_retention"]).clip(lower=0.005, upper=0.30)
    df["ltv_proxy"] = df["gross_margin"] / df["implied_churn"]
    df["ltv_proxy"] = df["ltv_proxy"].clip(lower=0.01, upper=500)

    # ------------------------------------------------------------------ #
    # LTV/CAC ratio
    # ------------------------------------------------------------------ #
    df["ltv_cac_ratio"] = df["ltv_proxy"] / df["cac_proxy"]

    # ------------------------------------------------------------------ #
    # CAC efficiency (S&M intensity)
    # ------------------------------------------------------------------ #
    df["cac_efficiency"] = df["s_and_m"] / df["revenue"].replace(0, np.nan)

    # ------------------------------------------------------------------ #
    # Trend features — rolling linear slope
    # ------------------------------------------------------------------ #
    df["ltv_trend"] = _rolling_slope(df["ltv_proxy"], TREND_WINDOW)
    df["cac_trend"] = _rolling_slope(df["cac_proxy"], TREND_WINDOW)
    df["ratio_trend"] = _rolling_slope(df["ltv_cac_ratio"], TREND_WINDOW)
    df["margin_trend"] = _rolling_slope(df["gross_margin"], TREND_WINDOW)

    # ------------------------------------------------------------------ #
    # Merge growth data if available
    # ------------------------------------------------------------------ #
    if not growth_df.empty:
        keep_cols = ["date"]
        for col in ["revenueGrowth", "grossProfitGrowth"]:
            if col in growth_df.columns:
                keep_cols.append(col)
        if len(keep_cols) > 1:
            growth_sub = growth_df[keep_cols].copy()
            df = pd.merge(df, growth_sub, on="date", how="left")

    return df.dropna(subset=["ltv_cac_ratio", "cac_proxy", "ltv_proxy"])


# ====================================================================== #
# Helpers
# ====================================================================== #

def _rolling_slope(series: pd.Series, window: int = 4) -> pd.Series:
    """
    Compute rolling OLS slope over *window* periods.
    Returns a Series of the same length with NaN padding for the warm-up.
    """
    slopes: list[float] = []
    vals = series.values.astype(float)
    x = np.arange(window, dtype=float)

    for i in range(len(vals)):
        if i < window - 1:
            slopes.append(np.nan)
            continue
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            slopes.append(np.nan)
        else:
            # Simple OLS slope via polyfit degree-1
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)

    return pd.Series(slopes, index=series.index)
