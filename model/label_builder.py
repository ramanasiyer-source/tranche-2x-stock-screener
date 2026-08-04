"""
label_builder.py — Step 3 of the Tranche 2x Quant Pipeline.

For each observation in features_historical.csv, computes the binary label:

  label = 1  if price(T + 12m) / price(T) >= 2.0   (doubled)
             AND price(T + 15m) / price(T) >= 1.5   (sustained — not a flash spike)

  label = 0  otherwise (includes bankruptcies, flat movers, and temporary spikes)

Also computes auxiliary columns:
  - return_12m:   actual 12-month forward return
  - return_15m:   actual 15-month forward return  
  - max_drawdown: max intra-period drawdown (path quality metric)
  - path_quality: 'Clean', 'Volatile', or 'Spike-and-Drop'

Output:
  - model/data/labeled_dataset.csv
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_FILE, LABELED_FILE, PRICE_CACHE_DIR,
    FEATURE_COLS, DOUBLING_THRESHOLD, SUSTAIN_THRESHOLD,
    LABEL_HORIZON_MONTHS, SUSTAIN_MONTHS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def _load_price_hist(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(PRICE_CACHE_DIR, f'{ticker}.parquet')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def _price_as_of(hist: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Closest available closing price at or BEFORE date."""
    sub = hist[hist.index <= date]
    if sub.empty:
        return None
    close_col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    val = sub[close_col].iloc[-1]
    return float(val) if pd.notna(val) else None


def _price_after(hist: pd.DataFrame, date: pd.Timestamp, months: int) -> float | None:
    """
    Price approximately `months` months AFTER date.
    Uses the closest trading day at or after the target date.
    """
    target = date + pd.DateOffset(months=months)
    # Get price at or just after target (forward-looking)
    sub = hist[hist.index >= target]
    if sub.empty:
        # Try just before if we're at the end of history
        sub = hist[hist.index <= target]
        if sub.empty:
            return None
    close_col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    val = sub[close_col].iloc[0]
    return float(val) if pd.notna(val) else None


def _max_drawdown_in_period(hist: pd.DataFrame, start: pd.Timestamp, months: int) -> float | None:
    """
    Maximum peak-to-trough drawdown during the label period.
    Measures how rough the ride was on the way to the final price.
    """
    end = start + pd.DateOffset(months=months)
    sub = hist[(hist.index >= start) & (hist.index <= end)]
    if len(sub) < 5:
        return None
    close_col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    prices = sub[close_col].dropna()
    if prices.empty:
        return None
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return round(float(drawdown.min()) * 100, 2)  # negative number


def _path_quality(ret_12m: float, max_dd: float | None) -> str:
    """Classify the price trajectory quality."""
    if ret_12m >= 100:   # doubled
        if max_dd is None or max_dd > -20:
            return 'Clean Doubler'
        elif max_dd > -40:
            return 'Volatile Doubler'
        else:
            return 'Turbulent Doubler'
    elif ret_12m >= 50:
        return 'Strong Mover'
    elif ret_12m >= 0:
        return 'Modest Gain'
    else:
        return 'Loser'


def build_labels(force_refresh: bool = False) -> pd.DataFrame:
    """
    Join features with forward-looking return labels.
    Produces the final labeled_dataset.csv for model training.
    """
    if os.path.exists(LABELED_FILE) and not force_refresh:
        log.info(f'✅ Labels loaded from cache: {LABELED_FILE}')
        df = pd.read_csv(LABELED_FILE)
        log.info(f'   Positive rate: {df["label"].mean()*100:.1f}%  ({df["label"].sum()} doublers)')
        return df

    features_df = pd.read_csv(FEATURES_FILE)
    log.info(f'⚙️  Building labels for {len(features_df):,} observations...')

    results = []
    for i, row in features_df.iterrows():
        ticker = row['ticker']
        snap   = pd.Timestamp(row['snapshot_date'])

        if i % 100 == 0:
            print(f'  [{i:5,}/{len(features_df):,}] {ticker} @ {snap.date()}...', end='\r', flush=True)

        hist = _load_price_hist(ticker)
        if hist is None:
            continue

        p0  = _price_as_of(hist, snap)
        p12 = _price_after(hist, snap, LABEL_HORIZON_MONTHS)
        p15 = _price_after(hist, snap, SUSTAIN_MONTHS)

        if p0 is None or p0 <= 0 or p12 is None:
            continue

        ret_12m = round((p12 / p0 - 1) * 100, 2)
        ret_15m = round((p15 / p0 - 1) * 100, 2) if p15 else None
        max_dd  = _max_drawdown_in_period(hist, snap, LABEL_HORIZON_MONTHS)

        # ── Primary doubling label ────────────────────────────────────────────
        doubled   = (p12 / p0) >= DOUBLING_THRESHOLD
        sustained = (p15 / p0) >= SUSTAIN_THRESHOLD if p15 else False
        label = int(doubled and sustained)

        result = row.to_dict()
        result.update({
            'label':        label,
            'return_12m':   ret_12m,
            'return_15m':   ret_15m,
            'max_drawdown': max_dd,
            'path_quality': _path_quality(ret_12m, max_dd),
            'p0':           round(p0, 4),
            'p12':          round(p12, 4),
        })
        results.append(result)

    print()
    df = pd.DataFrame(results)
    df.to_csv(LABELED_FILE, index=False)

    pos_rate = df['label'].mean() * 100
    n_pos    = df['label'].sum()
    n_total  = len(df)

    log.info(f'✅ Labels saved: {n_total:,} observations → {LABELED_FILE}')
    log.info(f'   Doublers (label=1): {n_pos} ({pos_rate:.1f}%)')
    log.info(f'   Non-doublers (label=0): {n_total - n_pos}')

    # Summarize doubler characteristics
    doublers = df[df['label'] == 1]
    if len(doublers) > 0:
        log.info(f'\n📊 Doubler snapshot:')
        log.info(f'   Median 12m return:  {doublers["return_12m"].median():.1f}%')
        log.info(f'   Median max drawdown: {doublers["max_drawdown"].median():.1f}%')
        log.info(f'   Path quality breakdown:')
        for q, cnt in doublers['path_quality'].value_counts().items():
            log.info(f'     {q}: {cnt}')

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build doubling labels')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()

    df = build_labels(force_refresh=args.refresh)

    print(f'\n📊 Label Distribution by Snapshot Date:')
    print(df.groupby('snapshot_date').agg(
        n_obs=('label', 'count'),
        n_doublers=('label', 'sum'),
        pos_rate=('label', lambda x: f'{x.mean()*100:.1f}%')
    ).to_string())
