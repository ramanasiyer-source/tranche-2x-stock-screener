"""
regime_clusterer.py — Step 5 of the Tranche 2x Quant Pipeline.

Takes only the POSITIVE LABEL observations (actual doublers) and
clusters them by their feature profiles using K-Means.

This reveals the distinct "archetypes" through which small-cap stocks
double — helping us understand the WHY, not just the WHAT.

Output:
  - model/outputs/model_regime.pkl        (trained KMeans model)
  - model/data/regime_labels.csv          (doublers with regime assignments)
  - Printed regime profile summary
"""

import os
import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LABELED_FILE, FEATURE_COLS, N_REGIMES,
    MODEL_REGIME_FILE, REGIME_LABELS_FILE, RANDOM_STATE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Human-readable regime labels (assigned after inspecting cluster profiles)
# These will be auto-assigned based on the dominant feature in each cluster.
REGIME_TEMPLATES = {
    'high_rev_growth':    '🚀 Earnings Compounder',
    'high_momentum':      '⚡ Momentum Breakout',
    'low_valuation':      '💎 Rerating Value Play',
    'high_rd_speculative':'🔬 Speculative Moonshot',
}


def _assign_regime_label(cluster_centroid: dict) -> str:
    """
    Heuristically name a cluster based on its centroid feature values.
    Uses the most distinctive feature (highest z-score) to name it.
    """
    indicators = {
        'Earnings Compounder':   cluster_centroid.get('rev_growth_yoy', 0) or 0,
        'Momentum Breakout':     cluster_centroid.get('momentum_6m', 0) or 0,
        'Rerating Value Play':   -1 * (cluster_centroid.get('ps_ratio', 50) or 50),  # lower P/S = more "value"
        'Speculative Moonshot':  cluster_centroid.get('short_ratio', 0) or 0,
    }
    best = max(indicators, key=indicators.get)
    emoji_map = {
        'Earnings Compounder':  '🚀 Earnings Compounder',
        'Momentum Breakout':    '⚡ Momentum Breakout',
        'Rerating Value Play':  '💎 Rerating Value Play',
        'Speculative Moonshot': '🔬 Speculative Moonshot',
    }
    return emoji_map[best]


def cluster_regimes(force_refresh: bool = False) -> pd.DataFrame:
    """
    K-Means clustering on historical doublers.
    Returns dataframe of doublers with their regime labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    if os.path.exists(REGIME_LABELS_FILE) and not force_refresh:
        log.info(f'✅ Regime labels loaded from cache: {REGIME_LABELS_FILE}')
        return pd.read_csv(REGIME_LABELS_FILE)

    df = pd.read_csv(LABELED_FILE)
    doublers = df[df['label'] == 1].copy()
    log.info(f'🔍 Clustering {len(doublers)} historical doublers into {N_REGIMES} regimes...')

    if len(doublers) < N_REGIMES * 3:
        log.warning(f'⚠️  Too few doublers ({len(doublers)}) for {N_REGIMES} clusters. Reducing to 2.')
        k = max(2, len(doublers) // 3)
    else:
        k = N_REGIMES

    X = doublers[FEATURE_COLS].copy()
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # ── Determine optimal k via inertia elbow (optional log output) ───────────
    inertias = []
    for test_k in range(2, min(k + 3, len(doublers) // 2)):
        km_test = KMeans(n_clusters=test_k, random_state=RANDOM_STATE, n_init=10)
        km_test.fit(X_scaled)
        inertias.append((test_k, km_test.inertia_))
    log.info('   Inertia by k: ' + ', '.join([f'k={ki}: {inr:.0f}' for ki, inr in inertias]))

    # ── Final clustering ───────────────────────────────────────────────────────
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    cluster_ids = km.fit_predict(X_scaled)
    doublers = doublers.copy()
    doublers['cluster_id'] = cluster_ids

    # ── Name each cluster ─────────────────────────────────────────────────────
    centroids_scaled = km.cluster_centers_
    centroids_df = pd.DataFrame(
        scaler.inverse_transform(centroids_scaled),
        columns=FEATURE_COLS
    )

    cluster_name_map = {}
    used_names = set()
    for cid in range(k):
        centroid = centroids_df.iloc[cid].to_dict()
        name = _assign_regime_label(centroid)
        # Avoid duplicate names
        if name in used_names:
            name = name + f' #{cid+1}'
        used_names.add(name)
        cluster_name_map[cid] = name

    doublers['regime'] = doublers['cluster_id'].map(cluster_name_map)

    # ── Save model & labels ───────────────────────────────────────────────────
    regime_model = {
        'kmeans':      km,
        'imputer':     imp,
        'scaler':      scaler,
        'name_map':    cluster_name_map,
        'centroids_df':centroids_df,
    }
    with open(MODEL_REGIME_FILE, 'wb') as f:
        pickle.dump(regime_model, f)

    doublers.to_csv(REGIME_LABELS_FILE, index=False)

    # ── Print regime profiles ─────────────────────────────────────────────────
    log.info('\n📊 REGIME PROFILES (medians across doublers in each cluster):')
    key_feats = ['rev_growth_yoy', 'momentum_6m', 'gross_margin_pct',
                 'ps_ratio', 'short_ratio', 'return_12m']

    for cid in range(k):
        cluster_data = doublers[doublers['cluster_id'] == cid]
        name = cluster_name_map[cid]
        log.info(f'\n  [{name}]  n={len(cluster_data)}')
        for feat in key_feats:
            if feat in cluster_data.columns:
                med = cluster_data[feat].median()
                log.info(f'    {feat:<28}: {med:.1f}')

    log.info(f'\n✅ Regime model saved: {MODEL_REGIME_FILE}')
    log.info(f'✅ Regime labels saved: {REGIME_LABELS_FILE}')

    return doublers


def assign_regime(features_dict: dict) -> str:
    """
    Assign a regime label to a new stock given its feature values.
    Used during scoring of current candidates.
    """
    if not os.path.exists(MODEL_REGIME_FILE):
        return 'Unknown'

    with open(MODEL_REGIME_FILE, 'rb') as f:
        model = pickle.load(f)

    km       = model['kmeans']
    imp      = model['imputer']
    scaler   = model['scaler']
    name_map = model['name_map']

    vals = [features_dict.get(f) for f in FEATURE_COLS]
    vals = [v if (v is not None and np.isfinite(float(v))) else np.nan for v in vals]
    X = np.array(vals, dtype=float).reshape(1, -1)
    X_imp    = imp.transform(X)
    X_scaled = scaler.transform(X_imp)
    cluster  = int(km.predict(X_scaled)[0])
    return name_map.get(cluster, 'Unknown')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()

    df = cluster_regimes(force_refresh=args.refresh)

    print(f'\n📊 Regime Distribution:')
    print(df['regime'].value_counts().to_string())

    print(f'\n📊 Top 3 Doublers Per Regime:')
    for regime, group in df.groupby('regime'):
        print(f'\n  {regime}:')
        top3 = group.nlargest(3, 'return_12m')[['ticker', 'snapshot_date', 'return_12m', 'regime']]
        print(top3.to_string(index=False))
