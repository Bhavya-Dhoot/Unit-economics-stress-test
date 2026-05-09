"""
SEC EDGAR client for fetching quarterly financial data.

Uses the SEC's free XBRL Company Facts API — no API key required.
Rate limit: 10 requests/second (enforced via sleep).
Requires a User-Agent header per SEC policy.

Data source hierarchy:
  1. Company Facts API: per-company, all XBRL facts
  2. Falls back gracefully if a tag isn't filed by a company

XBRL tag mapping to FMP-equivalent fields:
  FMP field                        → XBRL concept(s)
  revenue                          → Revenues, RevenueFromContractWithCustomerExcludingAssessedTax
  grossProfit                      → GrossProfit
  sellingAndMarketingExpenses      → SellingAndMarketingExpense
  sellingGeneralAndAdministrative  → SellingGeneralAndAdministrativeExpense
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent
_USER_AGENT = "LTVCACExperiment research@experiment.local"
_BASE_URL = "https://data.sec.gov"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_MIN_REQUEST_INTERVAL = 0.12  # ~8 req/sec, well under 10/sec limit


class EDGARClient:
    """Client for SEC EDGAR XBRL API."""

    def __init__(self, user_agent: str = _USER_AGENT):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json",
        })
        self._last_request_time: float = 0
        self._ticker_to_cik: dict[str, int] = {}
        self.total_requests: int = 0
        self._load_ticker_map()

    def _load_ticker_map(self) -> None:
        """Load ticker → CIK mapping from SEC."""
        try:
            self._rate_limit()
            resp = self._session.get(_TICKER_MAP_URL, timeout=30)
            self.total_requests += 1
            resp.raise_for_status()
            data = resp.json()
            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = int(entry.get("cik_str", 0))
                if ticker and cik:
                    self._ticker_to_cik[ticker] = cik
            logger.info("Loaded %d ticker→CIK mappings from SEC", len(self._ticker_to_cik))
        except Exception as e:
            logger.error("Failed to load SEC ticker map: %s", e)

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def get_cik(self, ticker: str) -> int | None:
        """Look up CIK for a ticker symbol."""
        return self._ticker_to_cik.get(ticker.upper())

    def fetch_company_facts(self, ticker: str) -> dict[str, Any] | None:
        """
        Fetch all XBRL facts for a company from SEC EDGAR.
        Returns the full JSON response or None on failure.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            logger.warning("No CIK found for ticker %s", ticker)
            return None

        url = f"{_BASE_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json"
        try:
            self._rate_limit()
            resp = self._session.get(url, timeout=30)
            self.total_requests += 1
            if resp.status_code == 404:
                logger.debug("No EDGAR data for %s (CIK %d)", ticker, cik)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("EDGAR request failed for %s: %s", ticker, e)
            return None

    def fetch_income_statements(self, ticker: str, quarters: int = 12) -> pd.DataFrame:
        """
        Extract quarterly income statement data from EDGAR Company Facts.

        Returns a DataFrame with columns matching FMP's format:
          date, revenue, grossProfit, sellingGeneralAndAdministrativeExpenses,
          sellingAndMarketingExpenses

        Handles diverse XBRL taxonomies across sectors:
          - Tech/Healthcare: typically report GrossProfit directly
          - Banks: may only report Revenues + NoninterestExpense
          - Energy: may use SalesRevenueNet, CostOfRevenue
        """
        facts = self.fetch_company_facts(ticker)
        if facts is None:
            return pd.DataFrame()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            logger.debug("No us-gaap facts for %s", ticker)
            return pd.DataFrame()

        # --- Extract each field ---
        # XBRL tags to try for each field (in priority order)
        # Expanded to cover diverse sector reporting standards
        field_tags = {
            "revenue": [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueGoodsNet",
                "SalesRevenueServicesNet",
                "InterestIncomeExpenseNet",  # Banks
                "TotalRevenuesAndOtherIncome",  # Insurance
                "RevenuesNetOfInterestExpense",  # Banks alt
            ],
            "grossProfit": [
                "GrossProfit",
            ],
            "costOfRevenue": [
                "CostOfRevenue",
                "CostOfGoodsAndServicesSold",
                "CostOfGoodsSold",
                "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
            ],
            "sellingGeneralAndAdministrativeExpenses": [
                "SellingGeneralAndAdministrativeExpense",
                "GeneralAndAdministrativeExpense",
                "NoninterestExpense",  # Banks
            ],
            "sellingAndMarketingExpenses": [
                "SellingAndMarketingExpense",
                "SellingExpense",
            ],
            "operatingExpenses": [
                "OperatingExpenses",
                "CostsAndExpenses",
            ],
        }

        field_series: dict[str, pd.Series] = {}

        for field_name, tag_list in field_tags.items():
            for tag in tag_list:
                if tag in us_gaap:
                    series = self._extract_quarterly_series(us_gaap[tag])
                    if not series.empty:
                        field_series[field_name] = series
                        break

        if "revenue" not in field_series:
            logger.debug("No revenue data for %s", ticker)
            return pd.DataFrame()

        # --- Combine into DataFrame ---
        df = pd.DataFrame(field_series)
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True)

        # --- Compute grossProfit from COGS if not directly reported ---
        if "grossProfit" not in df.columns or df["grossProfit"].isna().all():
            if "costOfRevenue" in df.columns:
                df["grossProfit"] = df["revenue"] - df["costOfRevenue"]
            elif "operatingExpenses" in df.columns:
                # Rough fallback: gross profit ≈ revenue - 60% of opex
                # (conservative estimate when only total opex is available)
                df["grossProfit"] = df["revenue"] - df["operatingExpenses"] * 0.6

        # --- Fallback: SGA from operating expenses if missing ---
        sga_col = "sellingGeneralAndAdministrativeExpenses"
        if sga_col not in df.columns or df[sga_col].isna().all():
            if "operatingExpenses" in df.columns and "costOfRevenue" in df.columns:
                # SGA ≈ opex - COGS (what's left after cost of revenue)
                df[sga_col] = df["operatingExpenses"] - df["costOfRevenue"]
                df[sga_col] = df[sga_col].clip(lower=0)

        # Clean up helper columns
        for col in ["costOfRevenue", "operatingExpenses"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Take the most recent N quarters
        df = df.tail(quarters).reset_index(drop=True)

        # Add symbol column for compatibility
        df["symbol"] = ticker

        return df

    def _extract_quarterly_series(self, concept: dict) -> pd.Series:
        """
        Extract quarterly (10-Q) values from a Company Facts concept.

        Returns a Series indexed by filing end-date with the numeric value.
        Filters to USD, quarterly filings only.
        """
        units = concept.get("units", {})
        usd_data = units.get("USD", [])

        if not usd_data:
            return pd.Series(dtype=float)

        quarterly_records = []
        for entry in usd_data:
            form = entry.get("form", "")
            # Include 10-Q (quarterly) filings
            if form not in ("10-Q", "10-Q/A"):
                continue

            end_date = entry.get("end")
            val = entry.get("val")
            if end_date is None or val is None:
                continue

            quarterly_records.append({
                "date": end_date,
                "value": float(val),
            })

        if not quarterly_records:
            return pd.Series(dtype=float)

        records_df = pd.DataFrame(quarterly_records)
        # Deduplicate: if multiple filings for same end-date, take the latest
        records_df = records_df.drop_duplicates(subset="date", keep="last")
        return records_df.set_index("date")["value"]
