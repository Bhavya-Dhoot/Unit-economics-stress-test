"""
Configuration for the LTV/CAC Decomposition Fragility experiment.

No API keys required — data is sourced from SEC EDGAR (free, public).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Sector universe (8 GICS-aligned sectors)
# ---------------------------------------------------------------------------
SECTORS: list[str] = [
    "Technology",
    "Healthcare",
    "Consumer Cyclical",
    "Financial Services",
    "Industrials",
    "Communication Services",
    "Consumer Defensive",
    "Energy",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_PATH = RESULTS_DIR / "edgar_data_cache.parquet"

# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKERS_PER_SECTOR = 30
DEFAULT_TARGET_OBSERVATIONS = 500_000
DEFAULT_CAC_SHOCK = 0.30
MONTE_CARLO_SEED = 42
TREND_WINDOW = 4                      # quarters for rolling slope
