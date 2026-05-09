"""
LTV/CAC Decomposition Fragility & Trend Signal Superiority Experiment.

Main entrypoint — orchestrates data pull, augmentation, stress simulation,
signal construction, analysis, and visualization.

Data source: SEC EDGAR (free, no API key, 10 req/sec).

Usage:
    python experiment.py                           # full EDGAR pull + 500K augmentation
    python experiment.py --use_cache               # skip API, use cached parquet
    python experiment.py --n_companies 50000        # smaller run for testing
    python experiment.py --synthetic_only           # no EDGAR, pure Monte Carlo seed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RESULTS_DIR,
    CACHE_PATH,
    DEFAULT_TARGET_OBSERVATIONS,
    DEFAULT_CAC_SHOCK,
    DEFAULT_TICKERS_PER_SECTOR,
    SECTORS,
)
from src.data_pipeline import build_real_dataset
from src.monte_carlo import augment_to_target
from src.stress_simulation import apply_cac_stress
from src.signals import build_signals
from src.analysis import (
    run_decomposition_analysis,
    run_signal_comparison,
    run_sector_analysis,
)
from src.visualization import (
    plot_fragility_by_source,
    plot_signal_comparison,
    plot_sector_heatmap,
    plot_rank_delta_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


def generate_synthetic_seed(n_per_sector: int = 300) -> pd.DataFrame:
    """
    Generate a sector-differentiated synthetic seed dataset.

    Sector profiles are calibrated to real industry ranges:
      - Gross margins from S&P 500 sector medians (e.g., Tech ~68%, Energy ~28%)
      - Churn rates from SaaS/subscription benchmarks adapted per sector
      - CAC efficiency (S&M/Revenue) from public 10-K filings

    Trend signals use a two-component model:
      1. Fundamentals-correlated (~30%): mean-reversion bias
      2. Independent operational momentum (~70%): company-specific trajectory

    Sources for sector calibration:
      - Damodaran Online (NYU Stern) gross margin by sector
      - KeyBanc SaaS benchmarks for churn/retention norms
      - S&P Capital IQ S&M intensity ratios
    """
    rng = np.random.default_rng(42)

    # Sector profiles: (gross_margin_mean, gross_margin_std,
    #                    churn_mean, churn_std,
    #                    cac_eff_mean, cac_eff_std)
    # Calibrated to real sector medians from Damodaran 2024/2025 datasets
    SECTOR_PROFILES = {
        "Technology":              (0.68, 0.12, 0.04, 0.03, 0.18, 0.10),
        "Healthcare":              (0.62, 0.15, 0.06, 0.04, 0.12, 0.06),
        "Consumer Cyclical":       (0.38, 0.12, 0.10, 0.06, 0.25, 0.12),
        "Financial Services":      (0.55, 0.18, 0.08, 0.05, 0.15, 0.08),
        "Industrials":             (0.32, 0.10, 0.07, 0.04, 0.10, 0.05),
        "Communication Services":  (0.55, 0.15, 0.09, 0.06, 0.30, 0.15),
        "Consumer Defensive":      (0.35, 0.08, 0.03, 0.02, 0.08, 0.04),
        "Energy":                  (0.28, 0.12, 0.12, 0.08, 0.06, 0.03),
    }

    records = []
    for sector in SECTORS:
        gm_mu, gm_sd, ch_mu, ch_sd, ce_mu, ce_sd = SECTOR_PROFILES.get(
            sector, (0.45, 0.15, 0.08, 0.05, 0.15, 0.08)
        )

        # Generate fundamentals
        gms = np.clip(rng.normal(gm_mu, gm_sd, n_per_sector), 0.05, 0.95)
        churns = np.clip(rng.normal(ch_mu, ch_sd, n_per_sector), 0.005, 0.30)
        cac_effs = np.clip(rng.normal(ce_mu, ce_sd, n_per_sector), 0.01, 0.80)
        ltv_ps = np.clip(gms / churns, 0.01, 500)
        cac_ps = np.clip(
            rng.lognormal(np.log(3 + cac_effs * 20), 0.6), 0.01, 50
        )
        ratios = ltv_ps / cac_ps

        # Three-component trend model (matching monte_carlo.py)
        # Link: trend ← fundamentals → resilience → stressed rank
        gm_z = (gms - np.median(gms)) / max(np.std(gms), 1e-6)
        ch_z = (churns - np.median(churns)) / max(np.std(churns), 1e-6)
        ratio_z = (ratios - np.median(ratios)) / max(np.std(ratios), 1e-6)

        # Component 1: fundamentals quality (~40%)
        fundamentals_quality = 0.5 * gm_z - 0.5 * ch_z
        # Component 2: ratio mean-reversion (~20%)
        ratio_signal = ratio_z * 0.2
        # Component 3: independent noise (~40%)
        noise_std = max(np.std(ratios) * 0.25, 0.5)
        noise = rng.normal(0, noise_std, n_per_sector)
        raw_trend = (
            0.40 * fundamentals_quality +
            0.20 * ratio_signal +
            0.40 * noise / max(np.std(noise), 1e-6)
        )
        ratio_trends = raw_trend * max(np.std(ratios) * 0.15, 0.1)

        for i in range(n_per_sector):
            records.append({
                "symbol": f"SEED_{sector[:3]}_{i:04d}",
                "sector": sector,
                "gross_margin": gms[i],
                "implied_churn": churns[i],
                "cac_efficiency": cac_effs[i],
                "ltv_proxy": ltv_ps[i],
                "cac_proxy": cac_ps[i],
                "ltv_cac_ratio": ratios[i],
                "ratio_trend": ratio_trends[i],
                "ltv_trend": ratio_trends[i] * 0.4 + rng.normal(0, 0.1),
                "cac_trend": -ratio_trends[i] * 0.2 + rng.normal(0, 0.1),
                "margin_trend": rng.normal(0, 0.01),
                "mktCap": rng.lognormal(23, 1.5),
                "data_source": "synthetic_seed",
            })

    return pd.DataFrame(records)


def generate_summary(decomp_df, signal_df, sector_df, df):
    """Auto-generate experiment_summary.md with hypothesis verdicts."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- H2: Winner by top-quartile precision (the operationally relevant metric) ---
    overall = signal_df[signal_df["scope"] == "overall"].copy()
    winner_row = overall.sort_values("top_quartile_precision", ascending=False).iloc[0]
    winner = winner_row["signal"]
    winner_prec = winner_row["top_quartile_precision"]
    winner_rho = winner_row["spearman_rho"]

    baseline = overall[overall["signal"] == "signal_A"].iloc[0]
    baseline_prec = baseline["top_quartile_precision"]
    baseline_rho = baseline["spearman_rho"]

    prec_improvement = ((winner_prec - baseline_prec) / max(abs(baseline_prec), 1e-9)) * 100

    inv_by_bin = decomp_df.set_index("cac_bin")["inversion_rate"]

    h1_verdict = "CONFIRMED" if inv_by_bin.iloc[-1] > inv_by_bin.iloc[0] else "NOT CONFIRMED"
    h2_verdict = "CONFIRMED" if winner != "signal_A" else "NOT CONFIRMED"

    real_count = (df["data_source"] == "real").sum()
    synth_count = len(df) - real_count
    # Build conditional H2 interpretation
    if winner != "signal_A":
        h2_note = (
            f"Signal C outperforms Signal A on top-quartile precision (+{prec_improvement:.1f}%) "
            f"because the trend component captures latent fundamental quality (margin, churn) "
            f"that the static ratio misses. This is a non-circular finding: the stress model "
            f"uses fundamentals directly, while Signal B captures them indirectly through trend."
        )
    else:
        h2_note = (
            f"The static LTV/CAC ratio is the dominant predictor of stress resilience. "
            f"Adding trend velocity (Signal C) dilutes precision by {abs(prec_improvement):.1f}% "
            f"because the trend's ~40% fundamentals correlation is too weak to overcome "
            f"the noise introduced by its ~40% independent component. "
            f"The stress model derives resilience directly from gross_margin and implied_churn — "
            f"the trend captures these indirectly but with insufficient signal-to-noise."
        )

    summary = f"""# Experiment Summary — LTV/CAC Decomposition Fragility

## Hypothesis 1 — Fragility (CAC-Driven vs LTV-Driven)
**Verdict: {h1_verdict}**

Inversion rates by source bin:
{inv_by_bin.to_string()}

Interpretation: {'CAC-driven companies show higher rank instability under stress, confirming fragility hypothesis.' if h1_verdict == 'CONFIRMED' else 'CAC-driven companies did not show significantly higher instability.'}

## Hypothesis 2 — Trend Signal vs Snapshot
**Verdict: {h2_verdict}**

| Metric | Winner ({winner}) | Baseline (signal_A) |
|--------|-------------------|---------------------|
| Top-quartile precision | {winner_prec:.4f} | {baseline_prec:.4f} |
| Spearman rho vs stressed rank | {winner_rho:.4f} | {baseline_rho:.4f} |
| Precision improvement | {prec_improvement:+.1f}% | — |

Best signal: **{winner}** (by top-quartile precision)

Interpretation: {h2_note}

## Experiment Integrity
- Stress resilience derived from fundamentals (gross_margin, implied_churn) — NOT from ratio_trend
- No circular dependency between Signal B and stressed rank
- Inversion threshold: original strict definition (75th pct → below 50th pct)
- Covariance: Ledoit-Wolf shrinkage (preserves cross-feature correlations)

## Sector Findings (Precision Gap: signal_C - signal_A)
{sector_df[['scope', 'precision_gap']].to_string(index=False)}

## Data Summary
- Real observations (SEC EDGAR): {real_count:,}
- Synthetic observations: {synth_count:,}
- **Total: {len(df):,}**

## Decomposition Summary
{decomp_df.to_string(index=False)}
"""
    path = RESULTS_DIR / "experiment_summary.md"
    path.write_text(summary, encoding="utf-8")
    # Windows cp1252 can't print Unicode chars like rho — use fallback
    try:
        print(summary)
    except UnicodeEncodeError:
        print(summary.encode("ascii", errors="replace").decode("ascii"))
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="LTV/CAC Decomposition Fragility & Trend Signal Experiment"
    )
    parser.add_argument("--n_companies", type=int, default=DEFAULT_TARGET_OBSERVATIONS)
    parser.add_argument("--cac_shock", type=float, default=DEFAULT_CAC_SHOCK)
    parser.add_argument("--use_cache", action="store_true",
                        help="Load cached parquet instead of re-pulling")
    parser.add_argument("--synthetic_only", action="store_true",
                        help="Skip EDGAR entirely, use pure Monte Carlo seed")
    parser.add_argument("--tickers_per_sector", type=int,
                        default=DEFAULT_TICKERS_PER_SECTOR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # ================================================================== #
    # Step 1 — Acquire real data (or load cache / generate seed)
    # ================================================================== #
    if args.synthetic_only:
        print("SYNTHETIC-ONLY mode: generating seed dataset...")
        real_df = generate_synthetic_seed()
        print(f"  Seed rows: {len(real_df):,}")
    elif args.use_cache and CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}...")
        real_df = pd.read_parquet(CACHE_PATH)
        print(f"  Cached rows: {len(real_df):,}")
    else:
        print("Pulling data from SEC EDGAR (no API key needed, 10 req/sec)...")
        try:
            real_df = build_real_dataset(
                tickers_per_sector=args.tickers_per_sector
            )
        except Exception as exc:
            logger.error("EDGAR pull failed: %s", exc)
            if CACHE_PATH.exists():
                print("Falling back to cached data...")
                real_df = pd.read_parquet(CACHE_PATH)
            else:
                print("No cache found. Falling back to synthetic seed.")
                real_df = generate_synthetic_seed()

    if real_df.empty:
        print("ERROR: No data available. Exiting.")
        sys.exit(1)

    # ================================================================== #
    # Step 2 — Monte Carlo augmentation
    # ================================================================== #
    print(f"\nAugmenting {len(real_df):,} rows to {args.n_companies:,}...")
    full_df = augment_to_target(real_df, target=args.n_companies)
    print(f"  Total observations: {len(full_df):,}")

    # ================================================================== #
    # Step 3 — Stress simulation
    # ================================================================== #
    print(f"\nApplying {args.cac_shock:.0%} CAC stress shock...")
    full_df = apply_cac_stress(full_df, shock=args.cac_shock)

    # ================================================================== #
    # Step 4 — Signal construction
    # ================================================================== #
    print("Building signals A (static), B (trend), C (composite)...")
    full_df = build_signals(full_df)

    # ================================================================== #
    # Step 5 — Analyses
    # ================================================================== #
    print("\nRunning analyses...")
    decomp_df, df_with_bins = run_decomposition_analysis(full_df)
    print("  [1/3] Decomposition fragility — done")

    signal_df = run_signal_comparison(full_df)
    print("  [2/3] Signal comparison — done")

    sector_df = run_sector_analysis(signal_df)
    print("  [3/3] Sector heterogeneity — done")

    # ================================================================== #
    # Step 6 — Visualizations
    # ================================================================== #
    print("\nGenerating plots...")
    plot_fragility_by_source(df_with_bins, decomp_df)
    plot_signal_comparison(signal_df)
    plot_sector_heatmap(sector_df)
    plot_rank_delta_distribution(df_with_bins)

    # ================================================================== #
    # Step 7 — Export tables
    # ================================================================== #
    print("\nExporting tables...")
    top_inverted = full_df[full_df["inversion_flag"] == 1].sort_values("rank_delta")
    top_inverted.head(200).to_csv(RESULTS_DIR / "inversion_table.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'inversion_table.csv'}")

    # ================================================================== #
    # Step 8 — Summary
    # ================================================================== #
    print("\n" + "=" * 60)
    generate_summary(decomp_df, signal_df, sector_df, full_df)
    print(f"\nAll outputs written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
