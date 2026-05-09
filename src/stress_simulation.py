"""
Stress simulation module — dual shock with fundamentals-based resilience.

INTEGRITY CONSTRAINT: The stress model must NOT read ratio_trend directly.
Signal B (ratio_trend velocity) must predict stressed outcomes INDIRECTLY
through its natural correlation with fundamentals, not through mechanical
injection. This prevents circular reasoning in Hypothesis 2.

Economic model:

CHANNEL 1 — CAC SHOCK (heterogeneous by CAC dependency):
  All companies face higher acquisition costs. CAC-driven companies
  absorb disproportionately larger increases because their low-cost
  channels are the most saturated and price-sensitive.

CHANNEL 2 — LTV EROSION (churn acceleration for CAC-driven):
  Companies built on cheap acquisition historically attract lower-quality
  customers who churn faster under market stress. This compresses LTV
  simultaneously with CAC inflation.

RESILIENCE — DERIVED FROM FUNDAMENTALS (NOT from ratio_trend):
  Companies with strong fundamentals (high gross margin, low churn)
  have structural resilience:
    - High margin → more buffer to absorb cost increases
    - Low churn → sticky customer base less affected by market stress
  
  ratio_trend CORRELATES with these fundamentals because improving
  companies tend to have improving margins/retention. But the stress
  model does NOT see the trend — it only sees the fundamentals.
  
  If Signal C still outperforms Signal A after this change, it means
  the trend captures latent fundamental information that the static
  ratio misses. That is a legitimate, non-circular finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_CAC_SHOCK, MONTE_CARLO_SEED


def apply_cac_stress(
    df: pd.DataFrame,
    shock: float = DEFAULT_CAC_SHOCK,
    random_seed: int = MONTE_CARLO_SEED,
) -> pd.DataFrame:
    """
    Apply dual shock (CAC + LTV) with fundamentals-based resilience.

    The resilience factor is derived from gross_margin and implied_churn,
    NOT from ratio_trend, to avoid circular reasoning with Signal B.
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    n = len(df)

    # --- Inputs ---
    cac_dep = (
        df["cac_source_pct"].fillna(0.5).values
        if "cac_source_pct" in df.columns else np.full(n, 0.5)
    )

    # ================================================================== #
    # RESILIENCE — derived from fundamentals ONLY
    #
    # gross_margin:   high margin → more buffer → more resilient
    # implied_churn:  low churn → sticky customers → more resilient
    #
    # Composite: normalise each to [0,1] via within-dataset percentile
    # rank, then average. This is scale-invariant and doesn't privilege
    # any particular sector's absolute values.
    #
    # CRITICAL: ratio_trend is NOT used here. This is what makes
    # the experiment non-circular.
    # ================================================================== #
    if "gross_margin" in df.columns and "implied_churn" in df.columns:
        gm = df["gross_margin"].fillna(df["gross_margin"].median())
        ch = df["implied_churn"].fillna(df["implied_churn"].median())

        # Percentile rank: higher is better for margin, lower is better for churn
        gm_pct = gm.rank(pct=True).values
        ch_pct = 1.0 - ch.rank(pct=True).values  # invert: low churn = high score

        # Composite resilience: [0, 1], 0.5 is neutral
        resilience = 0.5 * gm_pct + 0.5 * ch_pct
    else:
        # Fallback: no fundamentals available, use neutral resilience
        resilience = np.full(n, 0.5)

    # ================================================================== #
    # CHANNEL 1: CAC SHOCK
    #
    # base × (1 + 3 × cac_dep) - resilience discount + noise
    #
    # CAC multiplier range: [1.0, 4.0] → shock range [30%, 120%]
    # Resilience discount: ±30% of base → [-9%, +9%]
    # Noise: σ = 20% of base → ±6% typical
    # ================================================================== #
    cac_multiplier = 1.0 + 3.0 * cac_dep
    cac_resilience = shock * 2.0 * (resilience - 0.5)   # [-0.30, +0.30]
    cac_noise = rng.normal(0, shock * 0.20, size=n)

    cac_shock_applied = shock * cac_multiplier - cac_resilience + cac_noise
    cac_shock_applied = np.clip(cac_shock_applied, 0.05, None)

    # ================================================================== #
    # CHANNEL 2: LTV EROSION
    #
    # Base erosion: 40% × cac_dep^1.5 (non-linear, hits CAC-driven hard)
    # Resilience: high-margin, low-churn companies erode less
    # Noise: σ = 5%
    # ================================================================== #
    ltv_base_erosion = 0.40 * (cac_dep ** 1.5)
    ltv_resilience = 0.30 * (resilience - 0.5)           # [-0.15, +0.15]
    ltv_noise = rng.normal(0, 0.05, size=n)

    ltv_penalty = ltv_base_erosion - ltv_resilience + ltv_noise
    ltv_penalty = np.clip(ltv_penalty, 0.0, 0.50)

    # ================================================================== #
    # APPLY DUAL SHOCK
    # ================================================================== #
    df["cac_shock_applied"] = cac_shock_applied
    df["ltv_penalty_applied"] = ltv_penalty
    df["resilience_score"] = resilience
    df["cac_stressed"] = df["cac_proxy"] * (1.0 + cac_shock_applied)
    df["ltv_stressed"] = df["ltv_proxy"] * (1.0 - ltv_penalty)
    df["ratio_stressed"] = df["ltv_stressed"] / df["cac_stressed"]

    # --- Percentile ranks (within sector) ---
    df["rank_normal"] = df.groupby("sector")["ltv_cac_ratio"].rank(pct=True)
    df["rank_stressed"] = df.groupby("sector")["ratio_stressed"].rank(pct=True)
    df["rank_delta"] = df["rank_stressed"] - df["rank_normal"]

    # Inversion: top quartile (≥75th pct) fell to bottom half (<50th pct)
    # Using the ORIGINAL strict definition — not relaxed
    df["inversion_flag"] = (
        (df["rank_normal"] >= 0.75) & (df["rank_stressed"] < 0.50)
    ).astype(int)

    # --- Diagnostics ---
    n_inv = df["inversion_flag"].sum()
    inv_rate = n_inv / max(n, 1)
    cac_q = np.percentile(cac_shock_applied, [25, 50, 75])
    ltv_q = np.percentile(ltv_penalty, [25, 50, 75])

    print(f"  Stress diagnostics:")
    print(f"    Inversions: {n_inv:,} / {n:,} ({inv_rate:.2%})")
    print(f"    Rank delta: mean={df['rank_delta'].mean():.4f}, "
          f"std={df['rank_delta'].std():.4f}")
    print(f"    CAC shock:  P25={cac_q[0]:.1%}, P50={cac_q[1]:.1%}, P75={cac_q[2]:.1%}")
    print(f"    LTV erosion: P25={ltv_q[0]:.1%}, P50={ltv_q[1]:.1%}, P75={ltv_q[2]:.1%}")
    print(f"    Resilience source: gross_margin + implied_churn (NOT ratio_trend)")

    return df
