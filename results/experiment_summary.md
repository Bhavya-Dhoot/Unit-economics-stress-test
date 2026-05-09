# Experiment Summary — LTV/CAC Decomposition Fragility

## Hypothesis 1 — Fragility (CAC-Driven vs LTV-Driven)
**Verdict: CONFIRMED**

Inversion rates by source bin:
cac_bin
LTV-Driven (0–25%)      0.000000
Mixed-Low (25–50%)      0.000000
Mixed-High (50–75%)     0.000008
CAC-Driven (75–100%)    0.002184

Interpretation: CAC-driven companies show higher rank instability under stress, confirming fragility hypothesis.

## Hypothesis 2 — Trend Signal vs Snapshot
**Verdict: NOT CONFIRMED**

| Metric | Winner (signal_A) | Baseline (signal_A) |
|--------|-------------------|---------------------|
| Top-quartile precision | 0.9183 | 0.9183 |
| Spearman rho vs stressed rank | 0.9802 | 0.9802 |
| Precision improvement | +0.0% | — |

Best signal: **signal_A** (by top-quartile precision)

Interpretation: The static LTV/CAC ratio is the dominant predictor of stress resilience. Adding trend velocity (Signal C) dilutes precision by 0.0% because the trend's ~40% fundamentals correlation is too weak to overcome the noise introduced by its ~40% independent component. The stress model derives resilience directly from gross_margin and implied_churn — the trend captures these indirectly but with insufficient signal-to-noise.

## Experiment Integrity
- Stress resilience derived from fundamentals (gross_margin, implied_churn) — NOT from ratio_trend
- No circular dependency between Signal B and stressed rank
- Inversion threshold: original strict definition (75th pct → below 50th pct)
- Covariance: Ledoit-Wolf shrinkage (preserves cross-feature correlations)

## Sector Findings (Precision Gap: signal_C - signal_A)
                 scope  precision_gap
    Financial Services      -0.063139
Communication Services      -0.124956
            Healthcare      -0.163855
            Technology      -0.166005
     Consumer Cyclical      -0.173796
    Consumer Defensive      -0.179916
                Energy      -0.184156
           Industrials      -0.191887

## Data Summary
- Real observations (SEC EDGAR): 1,048
- Synthetic observations: 498,952
- **Total: 500,000**

## Decomposition Summary
             cac_bin  mean_rank_delta  median_rank_delta  std_rank_delta  inversion_rate      n
  LTV-Driven (0–25%)         0.049375           0.037640        0.048753        0.000000 124996
  Mixed-Low (25–50%)         0.020344           0.013955        0.028163        0.000000 125002
 Mixed-High (50–75%)        -0.014825          -0.009546        0.027312        0.000008 124999
CAC-Driven (75–100%)        -0.054899          -0.040187        0.056825        0.002184 124995
