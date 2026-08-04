"""
score_current.py — Step 6 of the Tranche 2x Quant Pipeline.

Applies the trained ensemble model to TODAY's small-cap universe
to produce a ranked list of stocks with 12-month doubling probabilities.

For each current candidate:
  - Computes current feature values (from cached yfinance data)
  - Runs ensemble prediction (LightGBM + XGBoost + Logistic Regression)
  - Assigns a regime label (via K-Means)
  - Computes top SHAP drivers (WHY this stock scored high)

Output:
  - model/outputs/current_scores.json     (full ranked results)
  - Printed leaderboard to console
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    UNIVERSE_FILE, FEATURE_COLS, CURRENT_SCORES_FILE,
    MODEL_LGBM_FILE, MODEL_XGB_FILE, MODEL_LR_FILE,
    SCALER_FILE, MODEL_REGIME_FILE, FUND_CACHE_DIR,
    PRICE_CACHE_DIR, MIN_MARKET_CAP_B, MAX_MARKET_CAP_B,
    LGBM_WEIGHT, XGB_WEIGHT, LR_WEIGHT
)
from feature_engineer import compute_features
from regime_clusterer import assign_regime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def _load_models():
    """Load all trained models from disk."""
    with open(MODEL_LGBM_FILE, 'rb') as f: lgbm = pickle.load(f)
    with open(MODEL_XGB_FILE,  'rb') as f: xgb  = pickle.load(f)
    with open(MODEL_LR_FILE,   'rb') as f: lr   = pickle.load(f)
    with open(SCALER_FILE,     'rb') as f: scaler = pickle.load(f)
    return lgbm, xgb, lr, scaler


def _get_shap_drivers(lgbm_model, X_row: np.ndarray, top_n: int = 4) -> list[dict]:
    """Compute SHAP values for a single observation and return top drivers."""
    try:
        import shap
        explainer   = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_row.reshape(1, -1))
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        drivers = []
        for i, (feat, val) in enumerate(zip(FEATURE_COLS, sv)):
            drivers.append({'feature': feat, 'shap': round(float(val), 4)})

        # Sort by absolute SHAP value descending
        drivers.sort(key=lambda x: abs(x['shap']), reverse=True)
        return drivers[:top_n]
    except Exception:
        return []


def _risk_label(doubling_prob: float, max_dd_hist: float | None) -> str:
    """Qualitative risk classification."""
    if doubling_prob >= 0.60:
        return 'High Conviction'
    elif doubling_prob >= 0.40:
        return 'Moderate Conviction'
    else:
        return 'Speculative'


def score_current_universe(today: str = None, force_refresh: bool = False) -> list[dict]:
    """
    Score all current universe tickers and return ranked results.
    """
    if today is None:
        today = pd.Timestamp.today().strftime('%Y-%m-%d')

    if os.path.exists(CURRENT_SCORES_FILE) and not force_refresh:
        log.info(f'✅ Current scores loaded from cache: {CURRENT_SCORES_FILE}')
        with open(CURRENT_SCORES_FILE) as f:
            return json.load(f)

    # ── Load models ───────────────────────────────────────────────────────────
    log.info('📦 Loading trained models...')
    from sklearn.impute import SimpleImputer
    lgbm_model, xgb_model, lr_model, scaler = _load_models()

    # ── Load universe ─────────────────────────────────────────────────────────
    universe = pd.read_csv(UNIVERSE_FILE)

    # Apply strict market cap filter for current scoring
    universe_filtered = universe[
        (universe['current_mcap_b'] >= MIN_MARKET_CAP_B) &
        (universe['current_mcap_b'] <= MAX_MARKET_CAP_B)
    ].copy()

    log.info(f'🔍 Scoring {len(universe_filtered)} stocks in $1B–$5B range as of {today}...')

    results = []
    imp = SimpleImputer(strategy='median')

    # Fit imputer on a representative set (use mean from training — load from labeled data)
    # Fallback: fit on current feature batch
    all_features_raw = []

    # First pass: collect all features
    for i, row in universe_filtered.iterrows():
        ticker = row['ticker']
        feat_dict = compute_features(ticker, today)
        if feat_dict is not None:
            all_features_raw.append(feat_dict)

    if not all_features_raw:
        log.error('❌ No features computed for any ticker. Check universe_builder.')
        return []

    feat_df = pd.DataFrame(all_features_raw)

    # Calculate sector-relative P/S for current universe
    if 'sector' in feat_df.columns:
        feat_df['sector_median_ps'] = feat_df.groupby('sector')['ps_ratio'].transform('median')
        feat_df['ps_ratio_relative'] = feat_df['ps_ratio'] / feat_df['sector_median_ps'].replace(0, np.nan)
        feat_df['ps_ratio_relative'] = feat_df['ps_ratio_relative'].fillna(1.0).clip(0, 10).round(3)
        feat_df.drop(columns=['sector_median_ps'], inplace=True)
    else:
        feat_df['ps_ratio_relative'] = 1.0

    X_raw   = feat_df[FEATURE_COLS].values

    # Impute and scale
    X_imp    = imp.fit_transform(X_raw)
    X_scaled = scaler.transform(X_imp)

    # ── Predictions ───────────────────────────────────────────────────────────
    lgbm_probs = lgbm_model.predict_proba(X_imp)[:, 1]
    xgb_probs  = xgb_model.predict_proba(X_imp)[:, 1]
    lr_probs   = lr_model.predict_proba(X_scaled)[:, 1]
    ens_probs  = LGBM_WEIGHT * lgbm_probs + XGB_WEIGHT * xgb_probs + LR_WEIGHT * lr_probs

    # ── Build output ──────────────────────────────────────────────────────────
    info_cache = {}
    for i, (feat_dict, lgbm_p, xgb_p, lr_p, ens_p, X_row) in enumerate(
        zip(all_features_raw, lgbm_probs, xgb_probs, lr_probs, ens_probs, X_imp)
    ):
        ticker = feat_dict['ticker']

        # Load cached info for name/sector
        info_path = os.path.join(FUND_CACHE_DIR, f'{ticker}.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
        else:
            info = {}

        # Regime assignment
        regime = assign_regime(feat_dict)

        # SHAP drivers
        shap_drivers = _get_shap_drivers(lgbm_model, X_row, top_n=4)

        # Feature snapshot (rounded for display)
        feat_display = {
            k: round(v, 3) if v is not None else None
            for k, v in feat_dict.items()
            if k in FEATURE_COLS
        }

        results.append({
            'ticker':               ticker,
            'name':                 info.get('shortName', ticker),
            'sector':               info.get('sector', 'Unknown'),
            'market_cap_b':         round(feat_dict.get('snap_mcap_b', 0), 2),
            'current_price':        round(feat_dict.get('snap_price', 0), 2),
            'fifty_two_week_high':  info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low':   info.get('fiftyTwoWeekLow'),
            'doubling_prob_12m':    round(float(ens_p), 4),
            'lgbm_prob':            round(float(lgbm_p), 4),
            'xgb_prob':             round(float(xgb_p), 4),
            'lr_prob':              round(float(lr_p), 4),
            'regime':               regime,
            'conviction':           _risk_label(float(ens_p), None),
            'top_drivers':          shap_drivers,
            'features':             feat_display,
            'scored_at':            today,
        })

    # ── Sort by doubling probability ──────────────────────────────────────────
    results.sort(key=lambda x: x['doubling_prob_12m'], reverse=True)

    # Add rank
    for rank, r in enumerate(results, 1):
        r['rank'] = rank

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(CURRENT_SCORES_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    log.info(f'✅ Scored {len(results)} stocks → {CURRENT_SCORES_FILE}')
    return results


def print_leaderboard(results: list[dict], top_n: int = 20) -> None:
    """Print a formatted leaderboard to console."""
    print(f'\n{"="*110}')
    print(f' 🏦 TRANCHE 2X LEADERBOARD — TOP {top_n} STOCKS (Doubling Probability Ranked)')
    print(f'{"="*110}')
    print(f' {"Rank":<5} {"Ticker":<7} {"Company":<28} {"Sector":<18} {"MCap":>7} {"Price":>8} {"2x Prob":>8} {"Regime":<28} {"Top Driver"}')
    print(f'{"-"*110}')

    for r in results[:top_n]:
        top_drv = r['top_drivers'][0]['feature'] if r['top_drivers'] else 'N/A'
        drv_shap = r['top_drivers'][0]['shap'] if r['top_drivers'] else 0
        drv_str = f'{top_drv} ({"+".join(str(drv_shap)) if drv_shap > 0 else str(drv_shap)})'
        print(
            f' #{r["rank"]:<4} '
            f'{r["ticker"]:<7} '
            f'{r["name"][:27]:<28} '
            f'{r["sector"][:17]:<18} '
            f'${r["market_cap_b"]:>5.1f}B '
            f'${r["current_price"]:>7.2f} '
            f'{r["doubling_prob_12m"]*100:>7.1f}% '
            f'{r["regime"]:<28} '
            f'{top_drv}'
        )
    print(f'{"="*110}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    results = score_current_universe(force_refresh=args.refresh)
    print_leaderboard(results, top_n=args.top)

    # Summary stats
    probs = [r['doubling_prob_12m'] for r in results]
    high_conv = [r for r in results if r['doubling_prob_12m'] >= 0.5]
    print(f'\n📊 Universe Stats:')
    print(f'   Total scored:       {len(results)}')
    print(f'   High conviction (≥50%): {len(high_conv)}')
    print(f'   Avg doubling prob:  {np.mean(probs)*100:.1f}%')
    print(f'   Max doubling prob:  {max(probs)*100:.1f}%  ({results[0]["ticker"]})')
