"""
portfolio_tracker.py — YoY Benchmark + Multi-Variant Mock Portfolio Engine

Two responsibilities:
  1. HISTORICAL BENCHMARK: For each year 2022–2025, show who the actual winners
     were, how the model would have ranked them, and how a top-10 portfolio
     would have performed vs the Russell 2000 (IWM) benchmark.

  2. CURRENT PORTFOLIOS: Build 5 parallel variant portfolios from today's model
     output — each representing a different investment hypothesis — and save them
     as a tracked manifest for ongoing monitoring.

Portfolio Variants:
  - Alpha:    Full ensemble model probability score
  - Momentum: Ranked by 6-month price momentum
  - Quality:  Ranked by composite quality score (FCF yield + gross margin - dilution)
  - Recovery: Ranked by stocks most beaten-down from 52w-high with positive fundamentals
  - Conviction: Only top 5 highest-probability picks, 2× weight

Output:
  - model/outputs/portfolio_benchmark.json   (historical YoY analysis)
  - model/outputs/portfolio_manifest.json    (current portfolios for live tracking)
"""

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LABELED_FILE, FEATURE_COLS, PRICE_CACHE_DIR, FUND_CACHE_DIR,
    CURRENT_SCORES_FILE, OUTPUTS_DIR,
    MODEL_LGBM_FILE, MODEL_XGB_FILE, MODEL_LR_FILE, SCALER_FILE,
    LGBM_WEIGHT, XGB_WEIGHT, LR_WEIGHT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

BENCHMARK_FILE  = os.path.join(OUTPUTS_DIR, 'portfolio_benchmark.json')
MANIFEST_FILE   = os.path.join(OUTPUTS_DIR, 'portfolio_manifest.json')

# Historical year windows: snapshot_date → (end_date, label_period)
YEAR_WINDOWS = [
    {'year': '2022', 'snapshot': '2022-01-03', 'end': '2023-01-03'},
    {'year': '2023', 'snapshot': '2023-01-03', 'end': '2024-01-02'},
    {'year': '2024', 'snapshot': '2024-01-02', 'end': '2025-01-02'},
    {'year': '2025', 'snapshot': '2024-07-01', 'end': '2025-07-01'},
]

N_PICKS = 10          # stocks per portfolio
NOTIONAL = 100_000    # $100K notional per portfolio


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_price(ticker: str) -> pd.DataFrame | None:
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
    sub = hist[hist.index <= date]
    if sub.empty:
        return None
    col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    val = sub[col].iloc[-1]
    return float(val) if pd.notna(val) else None


def _benchmark_return(ticker: str, start: str, end: str) -> float | None:
    """Compute total return for a benchmark ETF over the period."""
    hist = _load_price(ticker)
    if hist is None:
        # Download if not cached
        try:
            hist = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
        except Exception:
            return None

    p_start = _price_as_of(hist, pd.Timestamp(start))
    p_end   = _price_as_of(hist, pd.Timestamp(end) - pd.Timedelta(days=1))
    if p_start and p_end and p_start > 0:
        return round((p_end / p_start - 1) * 100, 2)
    return None


def _load_info(ticker: str) -> dict:
    path = os.path.join(FUND_CACHE_DIR, f'{ticker}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — HISTORICAL YoY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_benchmark() -> dict:
    """
    For each year window, identify actual doublers, retrospective model scores,
    and portfolio returns vs IWM benchmark.
    """
    log.info('\n📅 Building historical YoY benchmark...')

    # Load the full labeled dataset
    df = pd.read_csv(LABELED_FILE)

    # Load model for retroactive scoring
    try:
        from sklearn.impute import SimpleImputer
        with open(MODEL_LGBM_FILE, 'rb') as f: lgbm = pickle.load(f)
        with open(MODEL_XGB_FILE,  'rb') as f: xgb  = pickle.load(f)
        with open(MODEL_LR_FILE,   'rb') as f: lr   = pickle.load(f)
        with open(SCALER_FILE,     'rb') as f: scaler = pickle.load(f)
        models_loaded = True
    except Exception as e:
        log.warning(f'Could not load models: {e}. Skipping retroactive scores.')
        models_loaded = False

    benchmark_results = {}

    for window in YEAR_WINDOWS:
        year     = window['year']
        snap     = window['snapshot']
        end_date = window['end']

        log.info(f'\n  📊 Year {year}  ({snap} → {end_date})')

        year_df = df[df['snapshot_date'] == snap].copy()
        if year_df.empty:
            log.info(f'     No data for snapshot {snap}')
            continue

        n_total    = len(year_df)
        doublers   = year_df[year_df['label'] == 1]
        n_doublers = len(doublers)

        # ── Retroactive model scores ──────────────────────────────────────────
        if models_loaded:
            imp = SimpleImputer(strategy='median')
            X_raw = year_df[FEATURE_COLS]
            X_imp = imp.fit_transform(X_raw)
            X_sc  = scaler.transform(X_imp)

            lgbm_p = lgbm.predict_proba(X_imp)[:, 1]
            xgb_p  = xgb.predict_proba(X_imp)[:, 1]
            lr_p   = lr.predict_proba(X_sc)[:, 1]
            ens_p  = LGBM_WEIGHT * lgbm_p + XGB_WEIGHT * xgb_p + LR_WEIGHT * lr_p

            year_df = year_df.copy()
            year_df['model_score'] = ens_p
        else:
            year_df['model_score'] = 0.5

        year_df_sorted = year_df.sort_values('model_score', ascending=False)
        top_picks = year_df_sorted.head(N_PICKS)

        # ── Compute actual returns for top picks ──────────────────────────────
        picks_detail = []
        portfolio_returns = []
        for _, row in top_picks.iterrows():
            ticker = row['ticker']
            hist   = _load_price(ticker)
            info   = _load_info(ticker)

            if hist is not None:
                p0   = _price_as_of(hist, pd.Timestamp(snap))
                p_end= _price_as_of(hist, pd.Timestamp(end_date) - pd.Timedelta(days=1))
                ret  = round((p_end / p0 - 1) * 100, 2) if p0 and p_end and p0 > 0 else None
            else:
                ret = None

            if ret is not None:
                portfolio_returns.append(ret)

            picks_detail.append({
                'ticker':       ticker,
                'name':         info.get('shortName', ticker),
                'sector':       info.get('sector', 'Unknown'),
                'model_score':  round(float(row['model_score']), 3),
                'actual_return':ret,
                'doubled':      bool(row['label'] == 1),
                'return_12m_label': row.get('return_12m'),
            })

        # ── Benchmark returns ──────────────────────────────────────────────────
        iwm_ret = _benchmark_return('IWM', snap, end_date)   # Russell 2000
        spy_ret = _benchmark_return('SPY', snap, end_date)   # S&P 500

        # ── Portfolio return (equal weight) ───────────────────────────────────
        portfolio_return = round(float(np.mean(portfolio_returns)), 2) if portfolio_returns else None
        alpha_vs_iwm = round(portfolio_return - iwm_ret, 2) if portfolio_return and iwm_ret else None

        # ── Actual doublers in the year ───────────────────────────────────────
        actual_winners = []
        for _, row in doublers.nlargest(15, 'return_12m').iterrows():
            ticker = row['ticker']
            info   = _load_info(ticker)
            actual_winners.append({
                'ticker':      ticker,
                'name':        info.get('shortName', ticker),
                'return_12m':  row.get('return_12m'),
                'path_quality':row.get('path_quality', ''),
            })

        log.info(f'     Stocks in universe:    {n_total}')
        log.info(f'     Actual doublers:       {n_doublers} ({n_doublers/n_total*100:.1f}%)')
        log.info(f'     Portfolio return:      {portfolio_return}%')
        log.info(f'     IWM benchmark:         {iwm_ret}%')
        log.info(f'     Alpha vs IWM:          {alpha_vs_iwm}%')

        benchmark_results[year] = {
            'snapshot_date':      snap,
            'end_date':           end_date,
            'n_universe':         n_total,
            'n_doublers':         n_doublers,
            'doubler_rate_pct':   round(n_doublers / n_total * 100, 1),
            'top_picks':          picks_detail,
            'portfolio_return_pct': portfolio_return,
            'iwm_return_pct':     iwm_ret,
            'spy_return_pct':     spy_ret,
            'alpha_vs_iwm_pct':   alpha_vs_iwm,
            'actual_winners':     actual_winners,
        }

    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    log.info(f'\n✅ Historical benchmark saved: {BENCHMARK_FILE}')
    return benchmark_results


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — CURRENT PORTFOLIO VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def _quality_score(stock: dict) -> float:
    """Composite quality score: rewards high FCF yield + high margins, penalises dilution."""
    feats = stock.get('features', {})
    fcf    = feats.get('fcf_yield_pct')    or 0
    gm     = feats.get('gross_margin_pct') or 0
    dil    = feats.get('share_dilution_1y')or 0
    dte    = feats.get('debt_to_equity')   or 0
    return (fcf * 0.4) + (gm * 0.3) - (abs(dil) * 0.2) - (min(dte, 5) * 0.1)


def _recovery_score(stock: dict) -> float:
    """
    Recovery score: high score = beaten-down from 52w high BUT bouncing from low.
    The sweet spot is drawdown_from_52w_high < -30% AND recovery_from_52w_low > 20%.
    """
    feats = stock.get('features', {})
    dd    = feats.get('drawdown_from_52w_high') or 0    # negative = beaten down
    rec   = feats.get('recovery_from_52w_low')  or 0    # positive = bouncing
    dma   = feats.get('pct_vs_200dma')           or 0   # negative = below 200dma

    # Beaten down score (more negative = more beaten down = higher score)
    beaten_score = max(0, -dd)          # e.g. -40% dd → score = 40
    # Recovery confirmation (already bouncing off lows)
    bounce_score = max(0, rec - 10)     # only count if >10% off low
    # Penalise if too far above 200dma (not a recovery candidate)
    dma_penalty  = max(0, dma - 30)

    return beaten_score * 0.5 + bounce_score * 0.4 - dma_penalty * 0.1


def build_portfolio_variants(force_refresh: bool = False) -> dict:
    """
    Build 5 portfolio variants from today's current_scores.json.
    Each variant is a different lens on the same universe.
    """
    if os.path.exists(MANIFEST_FILE) and not force_refresh:
        log.info(f'✅ Portfolio manifest loaded from cache: {MANIFEST_FILE}')
        with open(MANIFEST_FILE) as f:
            return json.load(f)

    if not os.path.exists(CURRENT_SCORES_FILE):
        log.error('❌ current_scores.json not found. Run score_current.py first.')
        return {}

    with open(CURRENT_SCORES_FILE) as f:
        scores = json.load(f)

    today = datetime.today().strftime('%Y-%m-%d')
    log.info(f'\n💼 Building portfolio variants as of {today}...')
    log.info(f'   Universe size: {len(scores)} stocks')

    # Deduplicate by ticker (pick highest score if duplicate)
    seen = {}
    for s in scores:
        t = s['ticker']
        if t not in seen or s['doubling_prob_12m'] > seen[t]['doubling_prob_12m']:
            seen[t] = s
    scores_deduped = sorted(seen.values(), key=lambda x: x['doubling_prob_12m'], reverse=True)

    # ── Compute composite scores for each variant ─────────────────────────────
    for s in scores_deduped:
        s['_quality_score']  = _quality_score(s)
        s['_recovery_score'] = _recovery_score(s)
        s['_momentum_score'] = s.get('features', {}).get('momentum_6m') or -999

    # ── Build each variant ────────────────────────────────────────────────────
    def make_portfolio(name: str, description: str, sorted_stocks: list,
                       n: int = N_PICKS) -> dict:
        """Build a portfolio dict from top-N sorted stocks."""
        picks = sorted_stocks[:n]
        weight_per_stock = round(NOTIONAL / n, 2)

        holdings = []
        for rank, s in enumerate(picks, 1):
            holdings.append({
                'rank':          rank,
                'ticker':        s['ticker'],
                'name':          s['name'],
                'sector':        s['sector'],
                'regime':        s['regime'],
                'entry_price':   s['current_price'],
                'market_cap_b':  s['market_cap_b'],
                'notional':      weight_per_stock,
                'shares':        round(weight_per_stock / s['current_price'], 2)
                                 if s['current_price'] > 0 else 0,
                'model_prob':    s['doubling_prob_12m'],
                'top_driver':    s['top_drivers'][0]['feature']
                                 if s.get('top_drivers') else 'N/A',
                'features_snapshot': {
                    k: s.get('features', {}).get(k)
                    for k in ['rev_growth_yoy', 'momentum_6m', 'gross_margin_pct',
                              'fcf_yield_pct', 'drawdown_from_52w_high',
                              'recovery_from_52w_low', 'ps_ratio']
                },
            })

        return {
            'name':        name,
            'description': description,
            'created_at':  today,
            'n_picks':     n,
            'total_notional': NOTIONAL,
            'holdings':    holdings,
            'tracking': []   # daily NAV snapshots will be appended here
        }

    portfolios = {
        'alpha': make_portfolio(
            '🤖 Alpha (Full Model)',
            'Top 10 stocks by ensemble model probability (LightGBM + XGBoost + LogReg). '
            'The most data-driven portfolio — trusts all 20 features equally.',
            sorted(scores_deduped, key=lambda x: x['doubling_prob_12m'], reverse=True)
        ),

        'momentum': make_portfolio(
            '⚡ Momentum',
            'Top 10 stocks by 6-month price momentum. '
            'Hypothesis: stocks already moving have the strongest near-term continuation.',
            sorted(scores_deduped, key=lambda x: x['_momentum_score'], reverse=True)
        ),

        'quality': make_portfolio(
            '💎 Quality Compounder',
            'Top 10 by composite quality score: high FCF yield, high gross margin, '
            'low dilution, low leverage. Hypothesis: durable businesses re-rate over time.',
            sorted(scores_deduped, key=lambda x: x['_quality_score'], reverse=True)
        ),

        'recovery': make_portfolio(
            '🔄 Deep Recovery',
            'Top 10 stocks most beaten down from 52-week high but already bouncing. '
            'Hypothesis: fundamental survivors at depressed prices have highest return potential. '
            'Targets the UPST / HIMS / LMND archetype the base model underweighted.',
            sorted(scores_deduped, key=lambda x: x['_recovery_score'], reverse=True)
        ),

        'conviction': make_portfolio(
            '🎯 High Conviction (5-stock)',
            'Top 5 stocks only, doubled position size ($20K each). '
            'Concentrated bet on the model\'s highest-confidence picks.',
            sorted(scores_deduped, key=lambda x: x['doubling_prob_12m'], reverse=True),
            n=5
        ),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info(f'\n{"="*80}')
    log.info(f' 💼 CURRENT PORTFOLIO VARIANTS (as of {today})')
    log.info(f'{"="*80}')

    for key, port in portfolios.items():
        log.info(f'\n  {port["name"]}')
        log.info(f'  {port["description"][:70]}...')
        tickers = [h['ticker'] for h in port['holdings']]
        log.info(f'  Holdings: {", ".join(tickers)}')

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest = {
        'created_at':  today,
        'benchmark':   'IWM',   # Russell 2000 ETF
        'notional':    NOTIONAL,
        'n_picks':     N_PICKS,
        'portfolios':  portfolios,
    }

    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    log.info(f'\n✅ Portfolio manifest saved: {MANIFEST_FILE}')
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — PRETTY PRINT BENCHMARK TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_report(benchmark: dict) -> None:
    """Print a clean YoY benchmark scorecard to console."""
    print(f'\n{"="*90}')
    print(f'  📅 YEAR-OVER-YEAR BENCHMARK SCORECARD — TRANCHE 2X MODEL')
    print(f'{"="*90}')
    print(f'  {"Year":<6} {"Universe":>10} {"Doublers":>10} {"Rate":>6} '
          f'{"Portfolio":>12} {"IWM":>8} {"SPY":>8} {"Alpha":>8}')
    print(f'  {"-"*85}')

    for year, data in benchmark.items():
        port_ret = f'{data["portfolio_return_pct"]:+.1f}%' if data.get("portfolio_return_pct") is not None else 'N/A'
        iwm_ret  = f'{data["iwm_return_pct"]:+.1f}%'      if data.get("iwm_return_pct") is not None else 'N/A'
        spy_ret  = f'{data["spy_return_pct"]:+.1f}%'      if data.get("spy_return_pct") is not None else 'N/A'
        alpha    = f'{data["alpha_vs_iwm_pct"]:+.1f}%'    if data.get("alpha_vs_iwm_pct") is not None else 'N/A'

        print(f'  {year:<6} {data["n_universe"]:>10} {data["n_doublers"]:>10} '
              f'{data["doubler_rate_pct"]:>5.1f}% '
              f'{port_ret:>12} {iwm_ret:>8} {spy_ret:>8} {alpha:>8}')

    print(f'  {"="*85}')

    # Top picks detail per year
    for year, data in benchmark.items():
        print(f'\n  📊 {year} — Top 10 Model Picks vs Actual Outcome:')
        print(f'  {"Ticker":<7} {"Name":<28} {"Model Score":>12} {"Actual Return":>14} {"Doubled?":>9}')
        print(f'  {"-"*75}')
        for p in data.get('top_picks', []):
            ret = f'{p["actual_return"]:+.1f}%' if p.get('actual_return') is not None else 'N/A'
            dbl = '✅ YES' if p.get('doubled') else '❌ no'
            print(f'  {p["ticker"]:<7} {p.get("name","")[:27]:<28} '
                  f'{p["model_score"]:>12.3f} {ret:>14} {dbl:>9}')

        # Top actual winners
        print(f'\n  🏆 {year} — Actual Year Winners (retrospective):')
        for w in data.get('actual_winners', [])[:5]:
            print(f'     {w["ticker"]:<7} {w.get("name","")[:30]:<31} '
                  f'{w.get("return_12m", 0):+.1f}%   [{w.get("path_quality","")}]')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YoY Benchmark + Portfolio Builder')
    parser.add_argument('--refresh', action='store_true', help='Force refresh all outputs')
    parser.add_argument('--skip-benchmark', action='store_true', help='Skip historical benchmark')
    args = parser.parse_args()

    if not args.skip_benchmark:
        benchmark = build_historical_benchmark()
        print_benchmark_report(benchmark)

    manifest = build_portfolio_variants(force_refresh=args.refresh)

    print(f'\n{"="*80}')
    print(f'  💼 CURRENT PORTFOLIO VARIANTS SUMMARY')
    print(f'{"="*80}')
    for key, port in manifest.get('portfolios', {}).items():
        tickers = ', '.join(h['ticker'] for h in port['holdings'])
        print(f'\n  {port["name"]}')
        print(f'  {tickers}')

    print(f'\n  Files written:')
    print(f'    {BENCHMARK_FILE}')
    print(f'    {MANIFEST_FILE}')
