"""
train_model.py — Step 4 of the Tranche 2x Quant Pipeline.

Trains an ensemble classifier on the labeled historical dataset using
strict time-series cross-validation (no random splits, no lookahead).

Model Architecture:
  - LightGBM   (50% weight) — primary non-linear model
  - XGBoost    (30% weight) — secondary ensemble member
  - Logistic Regression (20% weight) — linear sanity check

Validation:
  - Expanding-window time-series CV (train on past, validate on next period)
  - Final held-out test: most recent snapshot (Jul 2024)

Outputs:
  - model/outputs/model_lgbm.pkl
  - model/outputs/model_xgb.pkl
  - model/outputs/model_lr.pkl
  - model/outputs/scaler.pkl
  - model/outputs/shap_values.csv
  - model/outputs/backtest_report.json
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LABELED_FILE, FEATURE_COLS, SNAPSHOT_DATES,
    MODEL_LGBM_FILE, MODEL_XGB_FILE, MODEL_LR_FILE,
    SCALER_FILE, SHAP_FILE, BACKTEST_REPORT_FILE,
    RANDOM_STATE, N_ESTIMATORS, LGBM_WEIGHT, XGB_WEIGHT, LR_WEIGHT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def _load_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Impute, clip and scale features. Returns X, y, dates."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X = df[FEATURE_COLS].copy()
    y = df['label'].astype(int)
    dates = pd.to_datetime(df['snapshot_date'])

    # Median imputation for missing values (no lookahead — fit per fold)
    imp = SimpleImputer(strategy='median')
    X_arr = imp.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=FEATURE_COLS, index=X.index)

    return X, y, dates


def _precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> float:
    """What fraction of the top-k scored stocks actually doubled?"""
    idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[idx].mean())


def _basket_return(y_true, y_prob, returns, k=10) -> float:
    """Equal-weight return of top-k picks vs all stocks."""
    idx = np.argsort(y_prob)[::-1][:k]
    return float(returns.iloc[idx].mean()) if len(idx) > 0 else 0.0


def train_and_evaluate(force_refresh: bool = False) -> dict:
    """Main training routine with time-series cross-validation."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, brier_score_loss
    import shap

    if all(os.path.exists(f) for f in [MODEL_LGBM_FILE, MODEL_XGB_FILE, MODEL_LR_FILE]) \
            and not force_refresh:
        log.info('✅ Trained models found in cache. Loading...')
        with open(MODEL_LGBM_FILE, 'rb') as f: lgbm_model = pickle.load(f)
        with open(MODEL_XGB_FILE,  'rb') as f: xgb_model  = pickle.load(f)
        with open(MODEL_LR_FILE,   'rb') as f: lr_model   = pickle.load(f)
        with open(SCALER_FILE,     'rb') as f: scaler     = pickle.load(f)
        return lgbm_model, xgb_model, lr_model, scaler

    df = pd.read_csv(LABELED_FILE)
    log.info(f'📂 Loaded labeled dataset: {len(df):,} observations')
    log.info(f'   Positive rate: {df["label"].mean()*100:.1f}%')

    # ── Ordered snapshot dates ─────────────────────────────────────────────────
    sorted_snaps = sorted(df['snapshot_date'].unique())
    log.info(f'   Snapshot dates available: {sorted_snaps}')

    # Hold out the LAST snapshot for final test
    test_snap  = sorted_snaps[-1]
    train_snaps = sorted_snaps[:-1]
    log.info(f'   Test (held-out): {test_snap}')
    log.info(f'   Train snapshots: {train_snaps}')

    # ── Cross-validation folds (expanding window) ────────────────────────────
    fold_results = []

    for fold_idx in range(1, len(train_snaps)):
        train_dates = train_snaps[:fold_idx]
        val_date    = train_snaps[fold_idx]

        train_df = df[df['snapshot_date'].isin(train_dates)]
        val_df   = df[df['snapshot_date'] == val_date]

        if len(train_df) < 30 or len(val_df) < 10:
            continue

        X_train_raw = train_df[FEATURE_COLS].copy()
        X_val_raw   = val_df[FEATURE_COLS].copy()
        y_train     = train_df['label'].astype(int)
        y_val       = val_df['label'].astype(int)

        # Impute and scale (fit on train only)
        imp = SimpleImputer(strategy='median')
        X_train_imp = imp.fit_transform(X_train_raw)
        X_val_imp   = imp.transform(X_val_raw)

        scaler_fold = StandardScaler()
        X_train_sc  = scaler_fold.fit_transform(X_train_imp)
        X_val_sc    = scaler_fold.transform(X_val_imp)

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # LightGBM
        lgbm_fold = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=5,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        lgbm_fold.fit(X_train_imp, y_train)
        lgbm_probs = lgbm_fold.predict_proba(X_val_imp)[:, 1]

        # XGBoost
        xgb_fold = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.03,
            max_depth=4,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
        )
        xgb_fold.fit(X_train_imp, y_train)
        xgb_probs = xgb_fold.predict_proba(X_val_imp)[:, 1]

        # Logistic Regression
        lr_fold = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        lr_fold.fit(X_train_sc, y_train)
        lr_probs = lr_fold.predict_proba(X_val_sc)[:, 1]

        # Ensemble
        ensemble_probs = (
            LGBM_WEIGHT * lgbm_probs +
            XGB_WEIGHT  * xgb_probs  +
            LR_WEIGHT   * lr_probs
        )

        # Metrics
        if y_val.sum() > 0:
            auc    = roc_auc_score(y_val, ensemble_probs)
            brier  = brier_score_loss(y_val, ensemble_probs)
            prec10 = _precision_at_k(y_val.values, ensemble_probs, k=min(10, len(y_val)))
        else:
            auc = brier = prec10 = float('nan')

        fold_results.append({
            'fold': fold_idx,
            'train_dates': list(train_dates),
            'val_date': val_date,
            'n_train': len(train_df),
            'n_val': len(val_df),
            'n_pos_val': int(y_val.sum()),
            'auc': round(auc, 4),
            'brier': round(brier, 4),
            'precision_at_10': round(prec10, 4),
        })

        log.info(
            f'   Fold {fold_idx}: val={val_date}  '
            f'AUC={auc:.3f}  Precision@10={prec10:.2f}  Brier={brier:.3f}'
        )

    # ── Full training on all non-test data ────────────────────────────────────
    log.info(f'\n🏋️  Training final model on all data except test ({test_snap})...')
    full_train_df = df[df['snapshot_date'] != test_snap]
    test_df       = df[df['snapshot_date'] == test_snap]

    X_train_full = full_train_df[FEATURE_COLS].copy()
    X_test_raw   = test_df[FEATURE_COLS].copy()
    y_train_full = full_train_df['label'].astype(int)
    y_test       = test_df['label'].astype(int)

    imp_final = SimpleImputer(strategy='median')
    X_train_imp_final = imp_final.fit_transform(X_train_full)
    X_test_imp        = imp_final.transform(X_test_raw)

    scaler_final = StandardScaler()
    X_train_sc_final = scaler_final.fit_transform(X_train_imp_final)
    X_test_sc        = scaler_final.transform(X_test_imp)

    pos_weight_final = (y_train_full == 0).sum() / max((y_train_full == 1).sum(), 1)

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=5,
        scale_pos_weight=pos_weight_final,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm_model.fit(X_train_imp_final, y_train_full)

    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=4,
        scale_pos_weight=pos_weight_final,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0,
    )
    xgb_model.fit(X_train_imp_final, y_train_full)

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr_model.fit(X_train_sc_final, y_train_full)

    # ── Test set evaluation ──────────────────────────────────────────────────
    lgbm_test = lgbm_model.predict_proba(X_test_imp)[:, 1]
    xgb_test  = xgb_model.predict_proba(X_test_imp)[:, 1]
    lr_test   = lr_model.predict_proba(X_test_sc)[:, 1]
    ens_test  = LGBM_WEIGHT * lgbm_test + XGB_WEIGHT * xgb_test + LR_WEIGHT * lr_test

    test_auc    = roc_auc_score(y_test, ens_test) if y_test.sum() > 0 else float('nan')
    test_brier  = brier_score_loss(y_test, ens_test) if y_test.sum() > 0 else float('nan')
    test_prec10 = _precision_at_k(y_test.values, ens_test, k=min(10, len(y_test)))

    log.info(f'\n🎯 TEST SET RESULTS ({test_snap}):')
    log.info(f'   AUC-ROC:        {test_auc:.4f}   (target ≥ 0.70)')
    log.info(f'   Precision@10:   {test_prec10:.4f}  (target ≥ 0.40)')
    log.info(f'   Brier Score:    {test_brier:.4f}  (target ≤ 0.15)')

    # ── SHAP values ───────────────────────────────────────────────────────────
    log.info('\n📐 Computing SHAP values...')
    explainer   = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_test_imp)

    # For binary classification, take class=1 SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    shap_df = pd.DataFrame(
        np.abs(sv).mean(axis=0).reshape(1, -1),
        columns=FEATURE_COLS,
        index=['mean_abs_shap']
    ).T.sort_values('mean_abs_shap', ascending=False)
    shap_df.to_csv(SHAP_FILE)

    log.info('\n🔑 Top 10 Most Important Features (SHAP):')
    for feat, val in shap_df.head(10).iterrows():
        log.info(f'   {feat:<28}  {val.iloc[0]:.5f}')

    # ── Feature importance from LightGBM ─────────────────────────────────────
    feat_imp = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': lgbm_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # ── Save models ───────────────────────────────────────────────────────────
    with open(MODEL_LGBM_FILE, 'wb') as f: pickle.dump(lgbm_model, f)
    with open(MODEL_XGB_FILE,  'wb') as f: pickle.dump(xgb_model, f)
    with open(MODEL_LR_FILE,   'wb') as f: pickle.dump(lr_model, f)
    with open(SCALER_FILE,     'wb') as f: pickle.dump(scaler_final, f)

    # ── Backtest report ───────────────────────────────────────────────────────
    report = {
        'cross_validation_folds': fold_results,
        'cv_avg_auc':          round(np.nanmean([f['auc'] for f in fold_results]), 4),
        'cv_avg_brier':        round(np.nanmean([f['brier'] for f in fold_results]), 4),
        'cv_avg_precision_10': round(np.nanmean([f['precision_at_10'] for f in fold_results]), 4),
        'test_set_date':       test_snap,
        'test_auc':            round(test_auc, 4),
        'test_brier':          round(test_brier, 4),
        'test_precision_at_10':round(test_prec10, 4),
        'top_features_shap':   shap_df.head(10).to_dict()['mean_abs_shap'],
        'top_features_lgbm':   feat_imp.set_index('feature')['importance'].head(10).to_dict(),
        'training_obs':        len(full_train_df),
        'test_obs':            len(test_df),
        'positive_rate_train': round(y_train_full.mean() * 100, 2),
        'positive_rate_test':  round(y_test.mean() * 100, 2),
    }

    with open(BACKTEST_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)

    log.info(f'\n✅ Models saved to {os.path.dirname(MODEL_LGBM_FILE)}')
    log.info(f'✅ Backtest report: {BACKTEST_REPORT_FILE}')

    return lgbm_model, xgb_model, lr_model, scaler_final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()

    train_and_evaluate(force_refresh=args.refresh)

    # Print report summary
    with open(BACKTEST_REPORT_FILE) as f:
        report = json.load(f)

    print(f'\n{"="*60}')
    print(' 🏦 TRANCHE 2X MODEL — BACKTEST SUMMARY')
    print(f'{"="*60}')
    print(f'  CV Avg AUC-ROC:        {report["cv_avg_auc"]:.4f}  (target ≥ 0.70)')
    print(f'  CV Avg Precision@10:   {report["cv_avg_precision_10"]:.4f}  (target ≥ 0.40)')
    print(f'  CV Avg Brier Score:    {report["cv_avg_brier"]:.4f}  (target ≤ 0.15)')
    print(f'  Test AUC-ROC:          {report["test_auc"]:.4f}')
    print(f'  Test Precision@10:     {report["test_precision_at_10"]:.4f}')
    print(f'\n  Training Observations: {report["training_obs"]:,}')
    print(f'  Test Observations:     {report["test_obs"]:,}')
    print(f'  Positive Rate (train): {report["positive_rate_train"]:.1f}%')
