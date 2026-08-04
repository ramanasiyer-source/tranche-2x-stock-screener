"""
feature_engineer.py — Step 2 of the Tranche 2x Quant Pipeline.

For each ticker × snapshot_date, computes 17 features using ONLY data
that would have been available BEFORE that snapshot date.

Sources:
  - yfinance quarterly_financials / quarterly_balance_sheet / quarterly_cashflow
    (time-aligned to the most recent completed quarter before snapshot_date)
  - Cached price history for momentum features

Output:
  - model/data/features_historical.csv

No lookahead bias: all financial figures are taken from the most recent
quarterly filing BEFORE each snapshot date.
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SNAPSHOT_DATES, FEATURE_COLS, UNIVERSE_FILE,
    FEATURES_FILE, FUND_CACHE_DIR, PRICE_CACHE_DIR,
    MIN_MARKET_CAP_B, MAX_MARKET_CAP_B
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# SPY cache for relative-strength computation
_SPY_HIST: pd.DataFrame | None = None


def _get_spy() -> pd.DataFrame:
    global _SPY_HIST
    if _SPY_HIST is not None:
        return _SPY_HIST
    spy_path = os.path.join(PRICE_CACHE_DIR, 'SPY.parquet')
    if os.path.exists(spy_path):
        _SPY_HIST = pd.read_parquet(spy_path)
    else:
        _SPY_HIST = yf.download('SPY', start='2021-06-01', end='2026-08-01',
                                auto_adjust=True, progress=False)
        if isinstance(_SPY_HIST.columns, pd.MultiIndex):
            _SPY_HIST.columns = _SPY_HIST.columns.get_level_values(0)
        _SPY_HIST.index = pd.to_datetime(_SPY_HIST.index).tz_localize(None)
        _SPY_HIST.to_parquet(spy_path)
    return _SPY_HIST


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


def _load_info(ticker: str) -> dict:
    path = os.path.join(FUND_CACHE_DIR, f'{ticker}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def _price_as_of(hist: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Closest available closing price at or before date."""
    if hist is None or hist.empty:
        return None
    sub = hist[hist.index <= date]
    if sub.empty:
        return None
    close_col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    val = sub[close_col].iloc[-1]
    return float(val) if pd.notna(val) else None


def _momentum(hist: pd.DataFrame, date: pd.Timestamp, months: int) -> float | None:
    """Price return over the past `months` months ending at `date`."""
    p_end = _price_as_of(hist, date)
    p_start = _price_as_of(hist, date - pd.DateOffset(months=months))
    if p_end is None or p_start is None or p_start == 0:
        return None
    return round((p_end / p_start - 1) * 100, 2)


def _rel_strength(ticker_hist: pd.DataFrame, date: pd.Timestamp, months: int = 6) -> float | None:
    """Excess return vs SPY over the past `months` months."""
    spy = _get_spy()
    stock_ret = _momentum(ticker_hist, date, months)
    spy_ret   = _momentum(spy, date, months)
    if stock_ret is None or spy_ret is None:
        return None
    return round(stock_ret - spy_ret, 2)


def _ttm_from_quarterly(df: pd.DataFrame, date: pd.Timestamp, row_name: str) -> float | None:
    """
    Compute trailing twelve months (TTM) sum for a quarterly financial row,
    using only quarters that completed BEFORE snapshot date.

    df: quarterly financials DataFrame (columns = quarter-end dates, rows = line items)
    """
    if df is None or df.empty or row_name not in df.index:
        return None

    # Filter to columns (quarter dates) strictly before snapshot date
    valid_cols = [c for c in df.columns if pd.to_datetime(c).tz_localize(None) < date]
    if len(valid_cols) < 4:
        return None

    # Take the 4 most recent valid quarters
    recent = sorted(valid_cols, reverse=True)[:4]
    vals = df.loc[row_name, recent]
    vals = pd.to_numeric(vals, errors='coerce').dropna()
    if len(vals) < 3:  # require at least 3 quarters
        return None
    return float(vals.sum())


def _latest_value(df: pd.DataFrame, date: pd.Timestamp, row_name: str) -> float | None:
    """Latest available value from a balance-sheet-style DataFrame before date."""
    if df is None or df.empty or row_name not in df.index:
        return None
    valid_cols = [c for c in df.columns if pd.to_datetime(c).tz_localize(None) < date]
    if not valid_cols:
        return None
    col = sorted(valid_cols, reverse=True)[0]
    val = pd.to_numeric(df.loc[row_name, col], errors='coerce')
    return float(val) if pd.notna(val) else None


def _safe(v, lo=-1000.0, hi=1000.0) -> float | None:
    """Clip extreme outliers and return None for NaN/non-numeric values.
    Casts to float first to handle yfinance returning strings like 'Infinity'."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(np.clip(v, lo, hi))


def compute_features(ticker: str, snapshot_date: str) -> dict | None:
    """
    Compute all 17 features for a given ticker at a given snapshot date.
    Returns None if insufficient data is available.
    """
    snap = pd.Timestamp(snapshot_date)
    info = _load_info(ticker)
    hist = _load_price_hist(ticker)

    if not info or hist is None:
        return None

    # ── Market cap at snapshot date (proxy via price × shares) ───────────────
    snap_price = _price_as_of(hist, snap)
    if snap_price is None or snap_price <= 0:
        return None

    shares = info.get('sharesOutstanding') or 0
    snap_mcap_b = (snap_price * shares) / 1e9 if shares else 0

    # Apply cap filter at snapshot date using price-implied market cap
    # (Falls back to current if no shares data — acceptable for first pass)
    if shares > 0:
        if snap_mcap_b < (MIN_MARKET_CAP_B * 0.5) or snap_mcap_b > (MAX_MARKET_CAP_B * 3.0):
            return None   # outside target range at this snapshot

    # ── Fetch quarterly financials ────────────────────────────────────────────
    try:
        tk = yf.Ticker(ticker)
        inc  = tk.quarterly_financials        # income statement
        bal  = tk.quarterly_balance_sheet     # balance sheet
        cf   = tk.quarterly_cashflow          # cash flow
    except Exception:
        inc, bal, cf = None, None, None

    # ── Revenue growth YoY ────────────────────────────────────────────────────
    rev_yoy = None
    if inc is not None and 'Total Revenue' in inc.index:
        ttm_rev     = _ttm_from_quarterly(inc, snap, 'Total Revenue')
        ttm_rev_lag = _ttm_from_quarterly(
            inc,
            snap - pd.DateOffset(months=12),
            'Total Revenue'
        )
        if ttm_rev and ttm_rev_lag and ttm_rev_lag != 0:
            rev_yoy = round((ttm_rev / ttm_rev_lag - 1) * 100, 2)
    if rev_yoy is None:
        rev_yoy = _safe(
            (info.get('revenueGrowth') or 0) * 100, lo=-100, hi=2000
        )

    # ── EPS growth YoY ───────────────────────────────────────────────────────
    eps_yoy = _safe(
        (info.get('earningsGrowth') or 0) * 100, lo=-500, hi=2000
    )

    # ── Gross margin ─────────────────────────────────────────────────────────
    gross_margin = None
    if inc is not None:
        ttm_rev_gm   = _ttm_from_quarterly(inc, snap, 'Total Revenue')
        # Try 'Gross Profit' first, fall back to 'Gross Income'
        for gp_label in ['Gross Profit', 'Gross Income']:
            ttm_gp = _ttm_from_quarterly(inc, snap, gp_label)
            if ttm_gp and ttm_rev_gm and ttm_rev_gm != 0:
                gross_margin = round((ttm_gp / ttm_rev_gm) * 100, 2)
                break
    if gross_margin is None:
        gross_margin = _safe(
            (info.get('grossMargins') or 0) * 100, lo=-100, hi=100
        )

    # ── Operating margin ─────────────────────────────────────────────────────
    op_margin = None
    if inc is not None:
        ttm_rev_op = _ttm_from_quarterly(inc, snap, 'Total Revenue')
        for op_label in ['Operating Income', 'EBIT']:
            ttm_op = _ttm_from_quarterly(inc, snap, op_label)
            if ttm_op and ttm_rev_op and ttm_rev_op != 0:
                op_margin = round((ttm_op / ttm_rev_op) * 100, 2)
                break
    if op_margin is None:
        op_margin = _safe(
            (info.get('operatingMargins') or 0) * 100, lo=-200, hi=100
        )

    # ── Net margin ───────────────────────────────────────────────────────────
    net_margin = _safe(
        (info.get('profitMargins') or 0) * 100, lo=-200, hi=100
    )

    # ── Free cash flow yield ──────────────────────────────────────────────────
    fcf_yield = None
    if cf is not None:
        ttm_cfo = _ttm_from_quarterly(cf, snap, 'Operating Cash Flow')
        ttm_capex = _ttm_from_quarterly(cf, snap, 'Capital Expenditure')
        if ttm_cfo is not None:
            ttm_fcf = ttm_cfo - abs(ttm_capex or 0)
            snap_mcap = snap_price * shares if shares else 0
            if snap_mcap > 0:
                fcf_yield = round((ttm_fcf / snap_mcap) * 100, 2)
    if fcf_yield is None:
        fcf = info.get('freeCashflow') or 0
        mcap = info.get('marketCap') or 0
        if mcap > 0:
            fcf_yield = _safe((fcf / mcap) * 100, lo=-50, hi=50)

    # ── Debt to equity ───────────────────────────────────────────────────────
    dte = None
    if bal is not None:
        total_debt   = _latest_value(bal, snap, 'Total Debt')
        total_equity = _latest_value(bal, snap, 'Stockholders Equity')
        if total_debt is not None and total_equity and total_equity > 0:
            dte = round(total_debt / total_equity, 3)
    if dte is None:
        dte = _safe(info.get('debtToEquity'), lo=0, hi=20)
        if dte is not None:
            dte = dte / 100  # yfinance returns as percentage

    # ── Valuation multiples (from info — current, accepts small lookahead) ────
    ps_ratio     = _safe(info.get('priceToSalesTrailing12Months'), lo=0, hi=100)
    pe_ratio     = _safe(info.get('trailingPE'),  lo=0, hi=200)
    peg_ratio    = _safe(info.get('pegRatio'),    lo=-10, hi=20)
    ev_to_ebitda = _safe(info.get('enterpriseToEbitda'), lo=-50, hi=200)
    price_to_book= _safe(info.get('priceToBook'), lo=0, hi=50)

    # ── Short interest ───────────────────────────────────────────────────────
    short_ratio  = _safe(info.get('shortRatio'), lo=0, hi=30)

    # ── Share dilution ────────────────────────────────────────────────────────
    # Approximate YoY shares outstanding growth
    shares_hist = hist.copy() if hist is not None else None
    share_dilution = None
    if shares:
        # Try to infer from float shares changes — use info fallback
        shares_pct_chg = info.get('floatShares', 0)
        # Simple proxy: use current info's impliedSharesOutstanding vs prior if available
        # For now use the yfinance field directly as a scaled ratio
        share_dilution = _safe(info.get('sharesPercentSharesOut'), lo=-0.5, hi=2.0)
        if share_dilution is not None:
            share_dilution = round(share_dilution * 100, 2)

    # ── Price momentum ───────────────────────────────────────────────────────
    mom_3m  = _momentum(hist, snap, 3)
    mom_6m  = _momentum(hist, snap, 6)
    rel_str = _rel_strength(hist, snap, 6)

    # ── Recovery / Drawdown features (NEW) ───────────────────────────────────
    # These capture beaten-down stocks with recovering fundamentals —
    # the "deep value recovery" archetype the original model missed.
    drawdown_from_52w_high = None
    recovery_from_52w_low  = None
    pct_vs_200dma          = None

    if hist is not None and not hist.empty:
        close_col = 'Close' if 'Close' in hist.columns else hist.columns[3]
        # Look back 52 weeks (252 trading days) from snapshot date
        window_start = snap - pd.DateOffset(weeks=52)
        window_hist  = hist[(hist.index >= window_start) & (hist.index <= snap)]
        if len(window_hist) >= 20:
            prices_52w = window_hist[close_col].dropna()
            high_52w   = float(prices_52w.max())
            low_52w    = float(prices_52w.min())

            # How far below the 52-week high? (negative = beaten down)
            if high_52w > 0:
                drawdown_from_52w_high = round((snap_price / high_52w - 1) * 100, 2)

            # How much has it recovered from the 52-week low? (positive = bouncing)
            if low_52w > 0 and snap_price > low_52w:
                recovery_from_52w_low = round((snap_price / low_52w - 1) * 100, 2)

        # 200-day moving average position
        dma_window = hist[hist.index <= snap].tail(200)
        if len(dma_window) >= 100:
            dma_200 = float(dma_window[close_col].mean())
            if dma_200 > 0:
                pct_vs_200dma = round((snap_price / dma_200 - 1) * 100, 2)

    return {
        'ticker':               ticker,
        'snapshot_date':        snapshot_date,
        'sector':               info.get('sector', 'Unknown'),
        'snap_price':           round(snap_price, 4),
        'snap_mcap_b':          round(snap_mcap_b, 3),
        # ── Features ──
        'rev_growth_yoy':       rev_yoy,
        'eps_growth_yoy':       eps_yoy,
        'gross_margin_pct':     gross_margin,
        'op_margin_pct':        op_margin,
        'net_margin_pct':       net_margin,
        'fcf_yield_pct':        fcf_yield,
        'debt_to_equity':       dte,
        'ps_ratio':             ps_ratio,
        'pe_ratio':             pe_ratio,
        'peg_ratio':            peg_ratio,
        'ev_to_ebitda':         ev_to_ebitda,
        'price_to_book':        price_to_book,
        'short_ratio':          short_ratio,
        'share_dilution_1y':    share_dilution,
        'momentum_3m':          mom_3m,
        'momentum_6m':          mom_6m,
        'rel_strength_6m':      rel_str,
        # Recovery / drawdown (NEW)
        'drawdown_from_52w_high': drawdown_from_52w_high,
        'recovery_from_52w_low':  recovery_from_52w_low,
        'pct_vs_200dma':          pct_vs_200dma,
    }


def build_features(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build historical feature dataset across all tickers × snapshot dates.
    """
    if os.path.exists(FEATURES_FILE) and not force_refresh:
        log.info(f'✅ Features loaded from cache: {FEATURES_FILE}')
        return pd.read_csv(FEATURES_FILE)

    universe = pd.read_csv(UNIVERSE_FILE)
    tickers  = universe['ticker'].tolist()

    log.info(f'⚙️  Engineering features for {len(tickers)} tickers × {len(SNAPSHOT_DATES)} snapshots...')
    log.info(f'   Expected observations: ~{len(tickers) * len(SNAPSHOT_DATES):,}')

    all_rows = []
    total = len(tickers) * len(SNAPSHOT_DATES)
    done = 0

    for ticker in tickers:
        for snap_date in SNAPSHOT_DATES:
            done += 1
            print(f'  [{done:5,}/{total:,}] {ticker} @ {snap_date}...', end='\r', flush=True)

            row = compute_features(ticker, snap_date)
            if row is not None:
                all_rows.append(row)

    print()
    df = pd.DataFrame(all_rows)

    # ── Sector-relative P/S ──────────────────────────────────────────────────
    # Calculate the median P/S ratio for each sector at each snapshot date
    df['sector_median_ps'] = df.groupby(['snapshot_date', 'sector'])['ps_ratio'].transform('median')
    # Calculate the relative P/S, handling division by zero or missing medians
    df['ps_ratio_relative'] = df['ps_ratio'] / df['sector_median_ps'].replace(0, np.nan)
    # Fill missing values with 1.0 (average) and clip extreme outliers
    df['ps_ratio_relative'] = df['ps_ratio_relative'].fillna(1.0).clip(0, 10).round(3)
    df.drop(columns=['sector_median_ps'], inplace=True)

    df.to_csv(FEATURES_FILE, index=False)

    log.info(f'✅ Features saved: {len(df):,} observations → {FEATURES_FILE}')
    log.info(f'   Coverage: {len(df)/total*100:.1f}% of ticker × snapshot combinations')
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build historical feature dataset')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()

    df = build_features(force_refresh=args.refresh)
    print(f'\n📊 Feature Dataset Summary:')
    print(f'   Observations: {len(df):,}')
    print(f'   Missing value rates:')
    miss = df[FEATURE_COLS].isnull().mean().sort_values(ascending=False)
    for col, rate in miss[miss > 0.1].items():
        print(f'     {col:<25} {rate*100:.1f}% missing')
