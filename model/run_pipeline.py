"""
run_pipeline.py — Master orchestrator for the Tranche 2x Quant Pipeline.

Runs all 6 steps in order:
  1. universe_builder    → downloads tickers + price history
  2. feature_engineer    → extracts 17 features at each historical snapshot
  3. label_builder       → computes doubling labels
  4. train_model         → trains LightGBM + XGBoost ensemble
  5. regime_clusterer    → K-Means clustering on historical doublers
  6. score_current       → scores today's universe → leaderboard

Usage:
  python run_pipeline.py            # run full pipeline (uses cache)
  python run_pipeline.py --refresh  # force re-download & re-train everything
  python run_pipeline.py --step 6   # run only from step 6 onwards
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def run_step(name: str, fn, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    log.info(f'\n{"─"*60}')
    log.info(f'▶  STEP: {name}')
    log.info(f'{"─"*60}')
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(f'✅ {name} completed in {elapsed:.1f}s')
        return result
    except Exception as e:
        log.error(f'❌ {name} FAILED: {e}')
        import traceback; traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description='Tranche 2x Quant Model Pipeline')
    parser.add_argument('--refresh', action='store_true',
                        help='Force re-download and re-train (ignores all caches)')
    parser.add_argument('--step', type=int, default=1,
                        help='Start from this step (1–6). Steps before are assumed cached.')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top stocks to display in final leaderboard')
    args = parser.parse_args()

    refresh = args.refresh
    start_step = args.step

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        🏦  TRANCHE 2X QUANT MODEL PIPELINE                   ║
║        Small-Cap ($1B–$5B) Doubling Probability Engine       ║
╠══════════════════════════════════════════════════════════════╣
║  Features:      17 (yfinance free tier)                      ║
║  Training data: 2022–2024 (6 snapshot windows)               ║
║  Model:         LightGBM + XGBoost + LogReg ensemble         ║
║  Regime:        K-Means (4 clusters)                         ║
╚══════════════════════════════════════════════════════════════╝
""")

    pipeline_start = time.time()

    # ── Step 1: Build Universe ────────────────────────────────────────────────
    if start_step <= 1:
        from universe_builder import build_universe
        run_step('Build Universe', build_universe, force_refresh=refresh)

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    if start_step <= 2:
        from feature_engineer import build_features
        run_step('Feature Engineering', build_features, force_refresh=refresh)

    # ── Step 3: Label Construction ────────────────────────────────────────────
    if start_step <= 3:
        from label_builder import build_labels
        run_step('Label Construction', build_labels, force_refresh=refresh)

    # ── Step 4: Model Training ────────────────────────────────────────────────
    if start_step <= 4:
        from train_model import train_and_evaluate
        run_step('Model Training & Backtest', train_and_evaluate, force_refresh=refresh)

    # ── Step 5: Regime Clustering ─────────────────────────────────────────────
    if start_step <= 5:
        from regime_clusterer import cluster_regimes
        run_step('Regime Clustering', cluster_regimes, force_refresh=refresh)

    # ── Step 6: Score Current Universe ──────────────────────────────────────────────────────
    if start_step <= 6:
        from score_current import score_current_universe, print_leaderboard
        results = run_step('Score Current Universe',
                           score_current_universe, force_refresh=refresh)
        print_leaderboard(results, top_n=args.top)

    # ── Step 7: Portfolio Tracker (YoY benchmark + variants) ─────────────────
    if start_step <= 7:
        from portfolio_tracker import build_historical_benchmark, build_portfolio_variants, print_benchmark_report
        benchmark = run_step('YoY Historical Benchmark', build_historical_benchmark)
        print_benchmark_report(benchmark)
        run_step('Portfolio Variants', build_portfolio_variants, force_refresh=refresh)

    total_time = time.time() - pipeline_start

    # ── Final Summary ─────────────────────────────────────────────────────────
    from config import BACKTEST_REPORT_FILE, CURRENT_SCORES_FILE
    print(f'\n{"="*70}')
    print(f' ✅  PIPELINE COMPLETE in {total_time/60:.1f} minutes')
    print(f'{"="*70}')

    if os.path.exists(BACKTEST_REPORT_FILE):
        with open(BACKTEST_REPORT_FILE) as f:
            rpt = json.load(f)
        print(f'\n 📊 MODEL PERFORMANCE:')
        print(f'    CV AUC-ROC:          {rpt.get("cv_avg_auc", "N/A")}  (target ≥ 0.70)')
        print(f'    CV Precision@10:     {rpt.get("cv_avg_precision_10", "N/A")}  (target ≥ 0.40)')
        print(f'    CV Brier Score:      {rpt.get("cv_avg_brier", "N/A")}  (target ≤ 0.15)')
        print(f'    Test AUC-ROC:        {rpt.get("test_auc", "N/A")}')
        print(f'    Test Precision@10:   {rpt.get("test_precision_at_10", "N/A")}')

    from config import CURRENT_SCORES_FILE, OUTPUTS_DIR
    PORTFOLIO_FILES = [
        os.path.join(OUTPUTS_DIR, 'portfolio_benchmark.json'),
        os.path.join(OUTPUTS_DIR, 'portfolio_manifest.json'),
    ]
    if os.path.exists(CURRENT_SCORES_FILE):
        with open(CURRENT_SCORES_FILE) as f:
            scores = json.load(f)
        print(f'\n 📈 CURRENT UNIVERSE:')
        print(f'    Stocks scored:          {len(scores)}')
        hc = [s for s in scores if s["doubling_prob_12m"] >= 0.5]
        print(f'    High conviction (≥50%): {len(hc)}')
        print(f'    Top pick:               {scores[0]["ticker"]} '
              f'({scores[0]["doubling_prob_12m"]*100:.1f}% probability)')
        print(f'    Regime:                 {scores[0]["regime"]}')

    print(f'\n Output files:')
    print(f'   model/outputs/current_scores.json       ← ranked leaderboard')
    print(f'   model/outputs/backtest_report.json      ← model performance')
    print(f'   model/outputs/shap_values.csv           ← feature importances')
    print(f'   model/outputs/portfolio_benchmark.json  ← YoY historical scorecard')
    print(f'   model/outputs/portfolio_manifest.json   ← 5 live portfolio variants')


if __name__ == '__main__':
    main()
