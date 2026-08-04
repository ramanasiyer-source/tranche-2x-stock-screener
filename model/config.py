"""
config.py — Central configuration for the Tranche 2x Quant Model.
All constants, snapshot dates, and feature definitions live here.
"""

# ── Universe Filters ──────────────────────────────────────────────────────────
MIN_MARKET_CAP_B = 1.0   # $1B floor  — ensures liquidity & data quality
MAX_MARKET_CAP_B = 5.0   # $5B ceiling — still enough room to 2x cleanly

# ── Label Thresholds ──────────────────────────────────────────────────────────
DOUBLING_THRESHOLD    = 2.0   # price(T+12m) / price(T) >= 2.0  → "doubled"
SUSTAIN_THRESHOLD     = 1.5   # price(T+15m) / price(T) >= 1.5  → "sustained"
LABEL_HORIZON_MONTHS  = 12    # forward window for primary label
SUSTAIN_MONTHS        = 15    # additional sustainability check month

# ── Training Snapshot Dates ───────────────────────────────────────────────────
# Each snapshot = one observation per stock.
# Features are computed from data BEFORE this date.
# Labels are computed from price data AFTER this date.
SNAPSHOT_DATES = [
    '2022-01-03',   # Fold 1 training
    '2022-07-01',   # Fold 2 training
    '2023-01-03',   # Fold 3 training
    '2023-07-03',   # Fold 4 training
    '2024-01-02',   # Fold 5 training
    '2024-07-01',   # Fold 6 validation / test set
]

# ── Feature Column Names ──────────────────────────────────────────────────────
# 21 features total: 17 original + 3 recovery/drawdown features + 1 sector relative feature
# All 17 features extractable from free yfinance data.
FEATURE_COLS = [
    # Growth
    'rev_growth_yoy',         # Revenue growth year-over-year (%)
    'eps_growth_yoy',         # EPS growth year-over-year (%)
    # Margins
    'gross_margin_pct',       # Gross margin (%)
    'op_margin_pct',          # Operating margin (%)
    'net_margin_pct',         # Net profit margin (%)
    # Cash & Leverage
    'fcf_yield_pct',          # Free cash flow / market cap (%)
    'debt_to_equity',         # Total debt / equity
    # Valuation
    'ps_ratio',               # Price-to-Sales (trailing 12m)
    'ps_ratio_relative',      # P/S relative to sector median at snapshot
    'pe_ratio',               # Trailing P/E (capped at 200 to handle outliers)
    'peg_ratio',              # PEG ratio (P/E / growth)
    'ev_to_ebitda',           # EV/EBITDA
    'price_to_book',          # Price-to-book value
    # Market / Sentiment
    'short_ratio',            # Short interest ratio (days to cover)
    'share_dilution_1y',      # YoY shares outstanding growth (%) — dilution risk
    # Price Momentum (computed from price history)
    'momentum_3m',            # 3-month price return (%)
    'momentum_6m',            # 6-month price return (%)
    'rel_strength_6m',        # Excess return vs S&P 500 over 6 months (%)
    # Recovery / Drawdown (NEW — captures deep-value recovery candidates)
    'drawdown_from_52w_high', # % below 52-week high (negative = beaten down)
    'recovery_from_52w_low',  # % bounce off 52-week low (positive = recovering)
    'pct_vs_200dma',          # % above/below 200-day moving average
]

# ── Model Settings ────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
N_REGIMES        = 4          # K-Means clusters for regime analysis
N_ESTIMATORS     = 500        # Trees per model
LGBM_WEIGHT      = 0.50       # Ensemble weight for LightGBM
XGB_WEIGHT       = 0.30       # Ensemble weight for XGBoost
LR_WEIGHT        = 0.20       # Ensemble weight for Logistic Regression

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, 'data')
CACHE_DIR         = os.path.join(BASE_DIR, 'cache')
PRICE_CACHE_DIR   = os.path.join(CACHE_DIR, 'prices')
FUND_CACHE_DIR    = os.path.join(CACHE_DIR, 'fundamentals')
OUTPUTS_DIR       = os.path.join(BASE_DIR, 'outputs')

for d in [DATA_DIR, CACHE_DIR, PRICE_CACHE_DIR, FUND_CACHE_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

UNIVERSE_FILE         = os.path.join(DATA_DIR, 'universe.csv')
FEATURES_FILE         = os.path.join(DATA_DIR, 'features_historical.csv')
LABELED_FILE          = os.path.join(DATA_DIR, 'labeled_dataset.csv')
REGIME_LABELS_FILE    = os.path.join(DATA_DIR, 'regime_labels.csv')
CURRENT_SCORES_FILE   = os.path.join(OUTPUTS_DIR, 'current_scores.json')
BACKTEST_REPORT_FILE  = os.path.join(OUTPUTS_DIR, 'backtest_report.json')
MODEL_LGBM_FILE       = os.path.join(OUTPUTS_DIR, 'model_lgbm.pkl')
MODEL_XGB_FILE        = os.path.join(OUTPUTS_DIR, 'model_xgb.pkl')
MODEL_LR_FILE         = os.path.join(OUTPUTS_DIR, 'model_lr.pkl')
MODEL_REGIME_FILE     = os.path.join(OUTPUTS_DIR, 'model_regime.pkl')
SCALER_FILE           = os.path.join(OUTPUTS_DIR, 'scaler.pkl')
SHAP_FILE             = os.path.join(OUTPUTS_DIR, 'shap_values.csv')

# ── Seed Universe: ~350 small/mid-cap tickers ($1B–$5B range) ────────────────
# Curated cross-sector list of stocks that were in or near the $1B–$5B range
# at various points between 2021–2025. Option A: slight survivorship bias,
# but sufficient for model validation. Upgrade to Russell 2000 historical
# constituents for production use.
SEED_TICKERS = [
    # ── Software / SaaS / Technology ─────────────────────────────────────
    'DUOL', 'DOCN', 'S',    'IOT',  'PATH', 'BILL', 'DBX',  'TOST', 'GTLB',
    'ZI',   'JAMF', 'NTNX', 'POWI', 'AEIS', 'AMBA', 'EXTR', 'IDCC', 'IRDM',
    'KTOS', 'LSCC', 'MGNI', 'MKSI', 'VRNS', 'SMAR', 'UPST', 'CWAN', 'EVBG',
    'RAMP', 'WEX',  'DT',   'FRSH', 'GLBE', 'MQ',   'NCNO', 'PCOR', 'PGNY',
    'SQSP', 'YEXT', 'ZETA', 'ACLS', 'APPN', 'AZPN', 'BAND', 'BRZE', 'CARG',
    'CDLX', 'CHGG', 'CPRT', 'CRSR', 'DAVA', 'DCBO', 'PRGS', 'SSTK', 'TRUP',
    'FOUR', 'APP',  'AVPT', 'INST', 'PRFT', 'WOLF', 'TASK', 'RPAY', 'OUST',
    'VNET', 'ZS',   'DDOG', 'ESTC', 'MDB',  'CFLT', 'SMAR', 'DOMO', 'ALTR',
    'PING', 'SPSC', 'QLYS', 'TENB', 'CYBE', 'NEWR', 'SUMO', 'MIST', 'LSPD',
    'TTEC', 'CSGS', 'EGHT', 'BLKB', 'QTWO', 'ALRM', 'PCTY', 'BL',   'PEGA',
    'GDDY', 'VOCS', 'CORT', 'RGEN', 'CNXN', 'CLFD', 'MIME', 'NABL', 'NLOK',
    'NTCT', 'NVEI', 'OMCL', 'OPEN', 'OPCH', 'OSPN', 'OTEX', 'PKOH', 'PLNT',

    # ── Healthcare / Biotech / Medtech ────────────────────────────────────
    'CYTK', 'HALO', 'NTRA', 'TXG',  'MEDP', 'INCY', 'AMPH', 'KRTX', 'ACAD',
    'RARE', 'ARQT', 'ACLX', 'BEAM', 'BHVN', 'CRNX', 'ENOV', 'EXEL', 'FOLD',
    'GKOS', 'HIMS', 'HRMY', 'IMVT', 'IRTC', 'ITCI', 'IOVA', 'KRYS', 'LGND',
    'MDGL', 'NBIX', 'NVCR', 'NVST', 'RPRX', 'RXRX', 'SRPT', 'TMDX', 'XNCR',
    'APLS', 'ARWR', 'ATRC', 'AUPH', 'AVIR', 'BCYC', 'CLDX', 'CNMD', 'CPRX',
    'CRSP', 'DXCM', 'EDIT', 'ELAN', 'ENTA', 'FGEN', 'FLGT', 'HOOK', 'IDYA',
    'IMCR', 'INSM', 'JAZZ', 'KNSL', 'KYMR', 'LNTH', 'LXRX', 'MRUS', 'NGMS',
    'NKTR', 'NTLA', 'NUVL', 'OCUL', 'PACB', 'PRAX', 'PTGX', 'RCKT', 'RETA',
    'RVMD', 'SAGE', 'SEER', 'SNDX', 'STOK', 'TARS', 'TELA', 'TGTX', 'TVTX',
    'TWST', 'VERA', 'VERV', 'VKTX', 'VRNA', 'RGEN', 'ACVA', 'ALGN', 'ALKS',
    'AMGN', 'ANIP', 'ARDX', 'BMRN', 'BNTX', 'CCXI', 'CDMO', 'CLVS', 'CNTA',
    'DCGO', 'ESPR', 'FATE', 'GILD', 'INMD', 'JAZZ', 'KDNY', 'LNTH', 'MDXG',
    'MNKD', 'MRNA', 'NARI', 'NVAX', 'OPCH', 'PBYI', 'PCRX', 'PMVP', 'PTLO',
    'RUBY', 'SIGA', 'SPRY', 'TBIO', 'TCDA', 'URGN', 'UTHR', 'VCNX', 'VSTM',

    # ── Consumer / Retail / Food & Bev ───────────────────────────────────
    'BROS', 'CAVA', 'WING', 'SHAK', 'BOOT', 'BIRK', 'CROX', 'ELF',  'SG',
    'CELH', 'DKNG', 'GRND', 'LESL', 'LMND', 'RENT', 'RVLV', 'SFIX', 'SKIN',
    'WOOF', 'WRBY', 'ARKO', 'BLMN', 'BJ',   'CAKE', 'CBRL', 'CHUY', 'CONN',
    'CPNG', 'DNUT', 'ELY',  'FAT',  'FRGI', 'GAIA', 'HVT',  'JACK', 'JOANN',
    'JBSS', 'KRUS', 'LOCO', 'LOVE', 'NDLS', 'NFLX', 'NTRA', 'PLAY', 'PLBY',
    'PRGO', 'PTLO', 'RICK', 'RRR',  'RUTH', 'SACH', 'SAFE', 'SCVL', 'SHOO',
    'SONO', 'SPWH', 'STKS', 'TXRH', 'UEIC', 'VSTO', 'WINA', 'ZUMZ',

    # ── Industrials / Clean Energy / Infra ───────────────────────────────
    'BLDP', 'PLUG', 'EVGO', 'RUN',  'CHPT', 'STEM', 'BE',   'FSLR', 'GNRC',
    'ARRY', 'AZRE', 'CLNE', 'ENPH', 'FLNC', 'GEVO', 'HASI', 'HYZN', 'NOVA',
    'SHLS', 'SPWR', 'ACHR', 'AEHR', 'AEVA', 'BLNK', 'CDRE', 'FLUX', 'JOBY',
    'LAZR', 'LICY', 'MARA', 'MVST', 'NKLA', 'PNTM', 'RIVN', 'SPCE', 'WKHS',
    'GEV',  'NEP',  'ARRY', 'CWEN', 'ORA',  'AY',   'MAXN', 'AMRC', 'PEGI',
    'VSLR', 'CLBT', 'BYRN', 'XPEV', 'NIO',  'LI',   'LCID', 'RIDE', 'FSR',
    'GOEV', 'HYLN', 'IDEX', 'KPLT', 'SLDP', 'MVST', 'AYRO', 'SOLO', 'NKLA',

    # ── Financial / Fintech ──────────────────────────────────────────────
    'AFRM', 'SOFI', 'UWMC', 'RKT',  'HCI',  'IIPR', 'INBK', 'LOAN', 'MFIN',
    'MLCO', 'MMSI', 'OPFI', 'PAYA', 'PNFP', 'PRAA', 'TREE', 'LCII',
    'PFSI', 'GHLD', 'LPLA', 'STEP', 'EVRI', 'PRAA', 'VRTS', 'HFWA', 'IIPR',
    'CUBI', 'HTBK', 'RNST', 'BUSE', 'NBTB', 'FSBC', 'VBTX', 'PPBI', 'CVB',
]
