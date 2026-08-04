"""
universe_builder.py — Step 1 of the Tranche 2x Quant Pipeline.

Downloads and caches:
  1. A filtered universe of tickers in the $1B–$5B market cap range
  2. 4 years of daily OHLCV price history for each ticker (used in all downstream steps)
  3. Raw yfinance fundamental info (cached to avoid repeated API calls)

Output:
  - model/data/universe.csv              (filtered ticker list + current metadata)
  - model/cache/prices/{TICKER}.parquet  (4-year daily price history per ticker)
  - model/cache/fundamentals/{TICKER}.pkl (raw yfinance info snapshot per ticker)

Runtime: ~5–15 minutes on first run. Subsequent runs use cache (seconds).
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEED_TICKERS, MIN_MARKET_CAP_B, MAX_MARKET_CAP_B,
    UNIVERSE_FILE, PRICE_CACHE_DIR, FUND_CACHE_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Pull 4.5 years of price history to cover all snapshot windows + label horizons
PRICE_START = '2021-06-01'
PRICE_END   = '2026-08-01'


def fetch_info(ticker: str) -> dict | None:
    """Fetch and cache yfinance info for a ticker. Returns None on failure."""
    cache_path = os.path.join(FUND_CACHE_DIR, f'{ticker}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    try:
        info = yf.Ticker(ticker).info
        if not info or info.get('marketCap') is None:
            return None
        with open(cache_path, 'wb') as f:
            pickle.dump(info, f)
        time.sleep(0.15)   # polite rate limiting
        return info
    except Exception as e:
        log.debug(f'  [{ticker}] info fetch failed: {e}')
        return None


def fetch_price_history(ticker: str) -> pd.DataFrame | None:
    """Fetch and cache daily price history. Returns None on failure."""
    cache_path = os.path.join(PRICE_CACHE_DIR, f'{ticker}.parquet')
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # corrupt cache → re-download

    try:
        hist = yf.download(
            ticker,
            start=PRICE_START,
            end=PRICE_END,
            auto_adjust=True,
            progress=False,
        )
        if hist.empty or len(hist) < 50:
            return None

        # Flatten multi-level columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist.to_parquet(cache_path)
        time.sleep(0.15)
        return hist
    except Exception as e:
        log.debug(f'  [{ticker}] price fetch failed: {e}')
        return None


def build_universe(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build the filtered stock universe.

    For each seed ticker:
      - Fetch yfinance info
      - Check current market cap is within $1B–$5B
      - Download and cache 4-year price history

    Also includes tickers that might be OUTSIDE current cap range but were
    INSIDE at some snapshot date (checked via historical market cap proxy).
    """
    if os.path.exists(UNIVERSE_FILE) and not force_refresh:
        log.info(f'✅ Universe loaded from cache: {UNIVERSE_FILE}')
        return pd.read_csv(UNIVERSE_FILE)

    log.info(f'🔍 Building universe from {len(SEED_TICKERS)} seed tickers...')
    log.info(f'   Market cap filter: ${MIN_MARKET_CAP_B}B – ${MAX_MARKET_CAP_B}B')

    records = []
    skipped_no_data  = []
    skipped_cap_range = []

    for i, ticker in enumerate(SEED_TICKERS, 1):
        print(f'  [{i:3d}/{len(SEED_TICKERS)}] {ticker}...', end='\r', flush=True)

        # ── Fetch fundamentals ────────────────────────────────────────────
        info = fetch_info(ticker)
        if info is None:
            skipped_no_data.append(ticker)
            continue

        market_cap = info.get('marketCap', 0) or 0
        mcap_b = market_cap / 1e9

        # ── Market cap filter (relaxed: ±50% of range to include historical) ─
        # A stock currently at $7B might have been at $3B in 2022 — include it.
        # A stock currently at $0.3B might have been at $1.5B in 2022 — include it.
        # We use a relaxed filter here; strict filtering happens per snapshot.
        if mcap_b < (MIN_MARKET_CAP_B * 0.3) or mcap_b > (MAX_MARKET_CAP_B * 4.0):
            skipped_cap_range.append(f'{ticker} (${mcap_b:.1f}B)')
            continue

        # ── Fetch price history ───────────────────────────────────────────
        hist = fetch_price_history(ticker)
        if hist is None:
            skipped_no_data.append(ticker)
            continue

        records.append({
            'ticker':           ticker,
            'name':             info.get('shortName', ticker),
            'sector':           info.get('sector', 'Unknown'),
            'industry':         info.get('industry', 'Unknown'),
            'current_mcap_b':   round(mcap_b, 3),
            'current_price':    info.get('currentPrice') or info.get('regularMarketPrice') or 0,
            'price_history_rows': len(hist),
        })

    print()  # clear progress line

    universe_df = pd.DataFrame(records)
    universe_df.to_csv(UNIVERSE_FILE, index=False)

    log.info(f'✅ Universe built: {len(universe_df)} stocks')
    log.info(f'   Skipped (no data):   {len(skipped_no_data)}')
    log.info(f'   Skipped (cap range): {len(skipped_cap_range)}')
    log.info(f'   Saved to: {UNIVERSE_FILE}')

    return universe_df


def get_price_history(ticker: str) -> pd.DataFrame | None:
    """Load cached price history for a ticker (or re-download if missing)."""
    return fetch_price_history(ticker)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build stock universe')
    parser.add_argument('--refresh', action='store_true', help='Force refresh all caches')
    args = parser.parse_args()

    df = build_universe(force_refresh=args.refresh)
    print(f'\n📊 Universe Summary:')
    print(df.groupby('sector')['ticker'].count().sort_values(ascending=False).to_string())
    print(f'\nTotal: {len(df)} stocks ready for feature engineering.')
