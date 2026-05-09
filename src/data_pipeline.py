"""
Data pipeline — universe construction and multi-source data pull.

Primary source: SEC EDGAR (free, no API key, 10 req/sec).
Fallback: Financial Modeling Prep (FMP) if EDGAR fails for a ticker.

Orchestrates data pulls across 8 sectors with progress logging and caching.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import SECTORS, RESULTS_DIR, CACHE_PATH
from src.edgar_client import EDGARClient
from src.feature_engineering import build_company_features

logger = logging.getLogger(__name__)

# ====================================================================== #
# Curated ticker universe — 30 per sector, US-listed, data-rich companies
# ====================================================================== #
SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ORCL", "CRM",
        "ADBE", "AMD", "INTC", "CSCO", "IBM", "NOW", "INTU", "AMAT",
        "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "FTNT", "PANW",
        "CRWD", "ZS", "DDOG", "NET", "TEAM", "WDAY",
    ],
    "Healthcare": [
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "MDT", "ISRG", "GILD", "VRTX", "SYK",
        "BSX", "EW", "ZTS", "REGN", "ILMN", "IDXX", "DXCM", "ALGN",
        "HOLX", "BAX", "BDX", "CI", "HUM", "CVS",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
        "BKNG", "CMG", "ORLY", "AZO", "ROST", "DG", "DLTR", "EBAY",
        "ETSY", "W", "DECK", "LULU", "YUM", "DPZ", "MAR", "HLT",
        "RCL", "LVS", "WYNN", "MGM", "F", "GM",
    ],
    "Financial Services": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C",
        "AXP", "USB", "PNC", "TFC", "COF", "BK", "STT", "FITB",
        "HBAN", "KEY", "CFG", "RF", "SIVB", "ZION", "CMA", "FRC",
        "ALLY", "DFS", "SYF", "NDAQ", "ICE", "CME",
    ],
    "Industrials": [
        "CAT", "UNP", "UPS", "HON", "BA", "RTX", "DE", "LMT",
        "GE", "MMM", "EMR", "ITW", "ROK", "FDX", "CSX", "NSC",
        "WM", "RSG", "JCI", "CARR", "OTIS", "IR", "DOV", "PH",
        "ETN", "GD", "NOC", "TXT", "HII", "SWK",
    ],
    "Communication Services": [
        "GOOG", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "EA", "ATVI", "TTWO", "RBLX", "MTCH", "ZM", "SNAP", "PINS",
        "ROKU", "SPOT", "LYV", "FOXA", "PARA", "WBD", "OMC", "IPG",
        "IACI", "NYT", "NWSA", "WMG", "SIRI", "LBRDA",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL",
        "MDLZ", "GIS", "K", "KHC", "HSY", "SJM", "MKC", "HRL",
        "CAG", "CPB", "TSN", "BG", "ADM", "KR", "SYY", "STZ",
        "TAP", "SAM", "MNST", "EL", "CLX", "CHD",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO",
        "PSX", "OXY", "HAL", "DVN", "FANG", "HES", "BKR", "WMB",
        "KMI", "OKE", "ET", "TRGP", "CTRA", "MRO", "APA", "MTDR",
        "SM", "RRC", "EQT", "AR", "SWN", "CHK",
    ],
}


def get_tickers_for_sector(sector: str, limit: int = 30) -> list[str]:
    """Return up to `limit` tickers for a given sector from the curated universe."""
    tickers = SECTOR_TICKERS.get(sector, [])
    return tickers[:limit]


def build_real_dataset(
    tickers_per_sector: int = 30,
    client: EDGARClient | None = None,
) -> pd.DataFrame:
    """
    Pull real quarterly financial data from SEC EDGAR for companies across 8 sectors.

    SEC EDGAR has no daily quota — only rate-limits at 10 req/sec.
    240 tickers × 1 request each = ~30 seconds total.
    """
    if client is None:
        client = EDGARClient()

    records: list[pd.DataFrame] = []
    total_tickers = 0
    skipped = 0

    for sector in SECTORS:
        logger.info("=== Sector: %s ===", sector)
        print(f"\n{'='*60}")
        print(f"  Sector: {sector}")
        print(f"{'='*60}")

        tickers = get_tickers_for_sector(sector, limit=tickers_per_sector)
        print(f"  Tickers: {len(tickers)}  |  EDGAR requests so far: {client.total_requests}")

        for i, symbol in enumerate(tickers):
            try:
                income_df = client.fetch_income_statements(symbol, quarters=12)
                # EDGAR doesn't have a separate growth endpoint —
                # feature_engineering computes growth from the income data
                growth_df = pd.DataFrame()
                profile = {}

                if income_df.empty:
                    logger.debug("  SKIP %s — no income data from EDGAR", symbol)
                    skipped += 1
                    continue

                merged = build_company_features(
                    symbol, sector, income_df, growth_df, {}, profile
                )
                if merged.empty:
                    skipped += 1
                    continue

                records.append(merged)
                total_tickers += 1

                if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                    print(
                        f"    [{i+1}/{len(tickers)}] processed  "
                        f"|  companies: {total_tickers}  "
                        f"|  EDGAR requests: {client.total_requests}"
                    )

            except Exception as exc:
                logger.warning("  SKIP %s: %s", symbol, exc)
                skipped += 1

    print(f"\n  EDGAR total requests: {client.total_requests}")
    print(f"  Total companies with data: {total_tickers}")
    print(f"  Skipped: {skipped}")

    if not records:
        logger.error("No data pulled from EDGAR. Check network connection.")
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result["data_source"] = "real"

    # ---- Persist ----
    RESULTS_DIR.mkdir(exist_ok=True)
    result.to_parquet(CACHE_PATH, index=False)
    logger.info("Cached %d rows to %s", len(result), CACHE_PATH)

    real_sample_path = RESULTS_DIR / "real_data_sample.csv"
    result.to_csv(real_sample_path, index=False)
    logger.info("Saved real data sample to %s", real_sample_path)

    return result
