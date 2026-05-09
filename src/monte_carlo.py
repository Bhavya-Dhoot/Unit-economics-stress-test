"""
Monte Carlo augmentation module.

Seeds from real FMP data distributions (per sector) and generates
synthetic company-quarter observations to reach the target count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MONTE_CARLO_SEED


# Feature columns used for distribution fitting
_FIT_FEATURES = [
    "ltv_proxy",
    "cac_proxy",
    "gross_margin",
    "implied_churn",
    "cac_efficiency",
]


def augment_to_target(
    real_df: pd.DataFrame,
    target: int = 500_000,
    random_seed: int = MONTE_CARLO_SEED,
) -> pd.DataFrame:
    """
    Augment real data to *target* total observations via calibrated
    Monte Carlo sampling.

    For each sector:
      1. Fit (log-mean, log-covariance) on real observations.
      2. Sample synthetic rows from multivariate log-normal.
      3. Generate trend features calibrated to real variance.
      4. Tag rows with data_source='synthetic'.

    Post-aggregation: compute cac_source_pct using sector medians.

    Args:
        real_df:     Real observations with all features computed.
        target:      Desired total observation count.
        random_seed: Reproducibility seed.

    Returns:
        Combined DataFrame (real + synthetic), with cac_source_pct added.
    """
    rng = np.random.default_rng(random_seed)
    sectors = real_df["sector"].unique()
    n_sectors = len(sectors)

    if n_sectors == 0:
        raise ValueError("real_df has no sector data — cannot augment.")

    synth_per_sector = max(1, (target - len(real_df)) // n_sectors)
    synth_records: list[pd.DataFrame] = []

    for sector in sectors:
        sdf = real_df[real_df["sector"] == sector]
        real_vals = sdf[_FIT_FEATURES].dropna()

        if len(real_vals) < 5:
            # Not enough real data — use global stats instead
            real_vals = real_df[_FIT_FEATURES].dropna()

        # ---- Fit sector distributions (log-space) ----
        log_vals = np.log(real_vals.clip(lower=1e-6))
        mu = log_vals.mean().values

        # Use Ledoit-Wolf shrinkage to preserve cross-feature correlations
        # (e.g., gross_margin ↔ ltv_proxy) while regularising small samples.
        # Diagonal-only would create unrealistic companies (e.g., 95% margin
        # with 30% churn — a combination absent in real markets).
        if len(log_vals) >= 10:
            try:
                from sklearn.covariance import LedoitWolf
                cov_stable = LedoitWolf().fit(log_vals.values).covariance_
            except ImportError:
                # Fallback: regularised full covariance
                cov = log_vals.cov().values
                cov_stable = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
        else:
            # Too few samples — use diagonal
            cov = log_vals.cov().values
            cov_stable = np.diag(np.diag(cov)) * 0.5

        # ---- Sample synthetic observations ----
        raw = rng.multivariate_normal(mu, cov_stable, size=synth_per_sector)
        synth = pd.DataFrame(np.exp(raw), columns=_FIT_FEATURES)

        # Derived metrics
        synth["ltv_cac_ratio"] = synth["ltv_proxy"] / synth["cac_proxy"]
        synth["sector"] = sector
        synth["symbol"] = [
            f"SYNTH_{sector[:3].upper()}_{i:06d}" for i in range(synth_per_sector)
        ]
        synth["data_source"] = "synthetic"
        synth["mktCap"] = np.nan

        # ---- CAC volatility multiplier (sector-calibrated) ----
        cac_mean = real_vals["cac_proxy"].mean()
        cac_std = real_vals["cac_proxy"].std()
        synth["cac_vol_mult"] = cac_std / max(cac_mean, 1e-6)

        # ---- Simulate trend features ----
        #
        # The ratio_trend has three components, designed so that Signal B
        # (trend velocity) has legitimate indirect predictive power for
        # stressed rank — WITHOUT the stress model reading ratio_trend.
        #
        # The link is: trend ← fundamentals → resilience → stressed rank
        #
        # 1. FUNDAMENTALS-QUALITY (~40%): correlated with gross_margin
        #    and implied_churn. Companies with high margins and low churn
        #    tend to have improving trends (they're executing well).
        #    This is the component that gives Signal B legitimate
        #    predictive power — it captures the same quality signal the
        #    stress model uses for resilience, but from the trend angle.
        #
        # 2. RATIO-CORRELATED (~20%): mild mean-reversion toward sector
        #    norms. Strong companies tend to keep improving, but weakly.
        #
        # 3. INDEPENDENT NOISE (~40%): company-specific trajectory changes
        #    unrelated to current fundamentals. Management changes,
        #    product launches, competitive shifts.
        #
        # If ratio_trend had ZERO correlation with fundamentals, Signal B
        # would have zero predictive power (correctly — random noise
        # shouldn't predict anything). If it had 100% correlation, it
        # would be redundant with Signal A. The ~40% fundamentals weight
        # is a realistic middle ground.

        gm_vals = synth["gross_margin"].values
        ch_vals = synth["implied_churn"].values
        ratio_vals = synth["ltv_cac_ratio"].values

        # Standardise fundamentals
        gm_z = (gm_vals - np.median(gm_vals)) / max(np.std(gm_vals), 1e-6)
        ch_z = (ch_vals - np.median(ch_vals)) / max(np.std(ch_vals), 1e-6)
        ratio_z = (ratio_vals - np.median(ratio_vals)) / max(np.std(ratio_vals), 1e-6)

        # Component 1: fundamentals quality (high margin + low churn = positive trend)
        fundamentals_quality = 0.5 * gm_z - 0.5 * ch_z

        # Component 2: ratio mean-reversion
        ratio_signal = ratio_z * 0.2

        # Component 3: independent noise
        noise_std = max(np.std(ratio_vals) * 0.25, 0.5)
        noise = rng.normal(0, noise_std, synth_per_sector)

        # Composite trend (weighted sum, re-scaled to natural units)
        raw_trend = (
            0.40 * fundamentals_quality +
            0.20 * ratio_signal +
            0.40 * noise / max(np.std(noise), 1e-6)
        )
        # Scale to match sector's ratio volatility
        synth["ratio_trend"] = raw_trend * max(np.std(ratio_vals) * 0.15, 0.1)

        # Sub-trends (directionally consistent with ratio_trend)
        synth["ltv_trend"] = (
            synth["ratio_trend"] * 0.4
            + rng.normal(0, max(synth["ltv_proxy"].std() * 0.08, 0.1), synth_per_sector)
        )
        synth["cac_trend"] = (
            -synth["ratio_trend"] * 0.2
            + rng.normal(0, max(synth["cac_proxy"].std() * 0.08, 0.1), synth_per_sector)
        )
        synth["margin_trend"] = rng.normal(0, 0.01, synth_per_sector)

        # Fill growth columns that real data might have
        for col in ["revenueGrowth", "grossProfitGrowth"]:
            if col not in synth.columns:
                synth[col] = np.nan

        synth_records.append(synth)

    # ---- Combine ----
    real_copy = real_df.copy()
    real_copy["data_source"] = "real"
    synth_df = pd.concat(synth_records, ignore_index=True)
    full_df = pd.concat([real_copy, synth_df], ignore_index=True)

    # ---- Post-aggregation: cac_source_pct ----
    full_df = _compute_cac_source_pct(full_df)

    return full_df.reset_index(drop=True)


def _compute_cac_source_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    CAC source decomposition using within-sector percentile rank.

    Uses inverted percentile rank of cac_efficiency within each sector:
      - Low cac_efficiency (low S&M spend relative to revenue) → high
        cac_source_pct → ratio strength driven by below-average CAC.
      - High cac_efficiency (high S&M spend) → low cac_source_pct →
        ratio strength driven by high LTV (margin/retention).

    This guarantees roughly uniform coverage across the 4 decomposition
    bins, unlike the median-deviation formula which clips half the data
    to zero.
    """
    # Inverted percentile rank: lowest cac_efficiency gets highest cac_source_pct
    df["cac_source_pct"] = 1.0 - df.groupby("sector")["cac_efficiency"].rank(pct=True)
    df["cac_source_pct"] = df["cac_source_pct"].clip(0, 1)

    # Also keep sector median for reference
    sector_medians = df.groupby("sector")["cac_efficiency"].median()
    df["sector_median_cac_eff"] = df["sector"].map(sector_medians)

    return df
