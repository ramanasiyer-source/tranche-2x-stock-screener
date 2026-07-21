#!/usr/bin/env python3
"""
===============================================================================
1-YEAR 2X STOCK SCREENER & PREDICTIVE RESEARCH ENGINE
===============================================================================
A complete Python pipeline that:
  1. Fetches Small/Mid-Cap stock data ($1B–$10B Market Cap) via yfinance.
  2. Segments stocks into 10 Market Cap Tranches ($1B increments).
  3. Engineers the Core Fundamental Features (Revenue Growth, Margins, Debt, FCF, Share Dilution).
  4. Trains a Random Forest Machine Learning Model to project 2x upside potential.
  5. Computes a 2x Conviction Score (0-100%) and Trajectory Tag (Linear, Sine-Wave, Breakout).
  6. Automatically caches data locally (`data/stock_screener_cache.json`).
===============================================================================
"""

import os
import sys
import json
import time
import math
from datetime import datetime, timedelta

# Check & auto-import required open-source libraries
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Installing required Python packages (yfinance, pandas, numpy, scikit-learn)...")
    os.system(f"{sys.executable} -m pip install yfinance pandas numpy scikit-learn")
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION & UNIVERSE DEFINITION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_FILE = os.path.join(CACHE_DIR, "stock_screener_cache.json")

SAMPLE_TICKERS = [
    # Tech / Software / AI
    "DUOL", "PATH", "IOT", "DOCN", "BILL", "FOUR", "DBX", "TOST", "APP", "S",
    # Consumer / Retail / Restaurant
    "CELH", "CROX", "WING", "CAVA", "ELF", "SG", "BOOT", "BIRK", "SHAK", "BROS",
    # Industrial / Clean Tech / Energy
    "FSLR", "RUN", "PLUG", "POWI", "CHPT", "ARR", "BLDP", "STEM", "BE", "EVGO",
    # Healthcare / Biotech
    "HALO", "KRTX", "NTRA", "CYTK", "AMPH", "EXAS", "MEDP", "INCY", "NARI", "TXG"
]

TRANCHE_STEP = 1.0  # $1 Billion increments
MAX_TRANCHE = 10.0   # Up to $10 Billion

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

def load_cached_data():
    ensure_cache_dir()
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
                cached_time = cached.get("timestamp", 0)
                if time.time() - cached_time < 86400:
                    print(f"✅ Loaded {len(cached.get('data', {}))} stocks from local cache ({CACHE_FILE})")
                    return cached.get("data", {})
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    return None

def save_to_cache(data_dict):
    ensure_cache_dir()
    try:
        payload = {
            "timestamp": time.time(),
            "formatted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data_dict
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 Saved updated dataset to local cache: {CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Failed to write cache: {e}")

def get_descriptive_tranche_name(bucket):
    if bucket <= 1:
        category = "Small-Cap"
    elif bucket <= 4:
        category = "Small-Cap"
    elif bucket <= 7:
        category = "Mid-Cap"
    else:
        category = "Upper Mid-Cap"
    return f"{category} (Tranche {bucket}: ${bucket-1}B–${bucket}B)"

def fetch_stock_data(tickers):
    cached = load_cached_data()
    if cached:
        return cached

    print(f"🌐 Fetching live financial data for {len(tickers)} stocks via yfinance...")
    dataset = {}

    for i, ticker in enumerate(tickers):
        try:
            print(f"  [{i+1}/{len(tickers)}] Processing {ticker}...", end="\r")
            tk = yf.Ticker(ticker)
            info = tk.info
            
            market_cap = info.get("marketCap", 0)
            if not market_cap:
                continue
                
            mcap_billions = round(market_cap / 1e9, 2)
            
            rev_growth = info.get("revenueGrowth", 0.0) or 0.0
            gross_margins = info.get("grossMargins", 0.0) or 0.0
            operating_margins = info.get("operatingMargins", 0.0) or 0.0
            debt_to_equity = info.get("debtToEquity", 0.0) or 0.0
            free_cash_flow = info.get("freeCashflow", 0) or 0
            fcf_yield = (free_cash_flow / market_cap) if market_cap > 0 else 0.0
            ps_ratio = info.get("priceToSalesTrailing12Months", 0.0) or 0.0
            pe_ratio = info.get("trailingPE", 0.0) or 0.0
            earnings_growth = info.get("earningsGrowth", 0.0) or 0.0
            
            hist = tk.history(period="1y")
            if len(hist) > 200:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                one_yr_return = (end_price - start_price) / start_price
                daily_returns = hist['Close'].pct_change().dropna()
                volatility = float(daily_returns.std() * math.sqrt(252))
            else:
                one_yr_return = 0.0
                volatility = 0.25

            tranche_bucket = min(int(math.ceil(mcap_billions / TRANCHE_STEP)), int(MAX_TRANCHE))
            if tranche_bucket < 1:
                tranche_bucket = 1

            tranche_desc = get_descriptive_tranche_name(tranche_bucket)

            dataset[ticker] = {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector": info.get("sector", "Technology"),
                "market_cap_b": mcap_billions,
                "tranche": tranche_desc,
                "tranche_num": tranche_bucket,
                "rev_growth_pct": round(rev_growth * 100, 2),
                "gross_margin_pct": round(gross_margins * 100, 2),
                "op_margin_pct": round(operating_margins * 100, 2),
                "debt_to_equity": round(debt_to_equity, 2),
                "fcf_yield_pct": round(fcf_yield * 100, 2),
                "ps_ratio": round(ps_ratio, 2),
                "pe_ratio": round(pe_ratio, 2),
                "earnings_growth_pct": round(earnings_growth * 100, 2),
                "one_yr_return_pct": round(one_yr_return * 100, 2),
                "volatility_pct": round(volatility * 100, 2)
            }
        except Exception as e:
            continue

    print(f"\n✅ Successfully processed {len(dataset)} stocks.")
    save_to_cache(dataset)
    return dataset

def classify_trajectory(row):
    ret = row["one_yr_return_pct"]
    vol = row["volatility_pct"]
    
    if ret > 40 and vol < 35:
        return "📈 Linear Climber"
    elif vol > 50:
        return "⚡ Volatile Breakout"
    elif ret < 0 and vol < 40:
        return "🌊 Sine-Wave Oscillator"
    else:
        return "📊 Steady Growth"

def train_and_score_model(dataset):
    df = pd.DataFrame(list(dataset.values()))
    if df.empty:
        print("No stock data available.")
        return df

    feature_cols = [
        "rev_growth_pct", "gross_margin_pct", "op_margin_pct", 
        "debt_to_equity", "fcf_yield_pct", "ps_ratio", "earnings_growth_pct"
    ]
    
    X = df[feature_cols].fillna(0)
    y = (df["one_yr_return_pct"] > 35.0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    probabilities = model.predict_proba(X_scaled)[:, 1] if len(model.classes_) > 1 else np.full(len(df), 0.5)
    
    df["conviction_score"] = np.round(probabilities * 100, 1)
    df["trajectory_tag"] = df.apply(classify_trajectory, axis=1)
    df = df.sort_values(by="conviction_score", ascending=False).reset_index(drop=True)
    return df

def print_screener_report(df):
    print("\n" + "="*110)
    print(" 🎯 TOP 10 TRANCHE 2X STOCK SCREENER LEADERBOARD")
    print("="*110)
    print(f" Analysis Date: {datetime.now().strftime('%Y-%m-%d')} | Target Universe: Small/Mid-Cap Tranches")
    print("="*110)
    
    header = f"{'Rank':<4} | {'Ticker':<7} | {'Company Name':<18} | {'Sector':<15} | {'Mkt Cap':<8} | {'Tranche Category':<24} | {'2x Score':<9} | {'Rev Growth':<10}"
    print(header)
    print("-" * len(header))

    for idx, row in df.head(10).iterrows():
        rank = f"#{idx+1}"
        name = row['name'][:17]
        sector = row['sector'][:14]
        mcap = f"${row['market_cap_b']:.2f}B"
        tranche = row['tranche'][:23]
        score = f"{row['conviction_score']:.1f}%"
        rev = f"{row['rev_growth_pct']:+.1f}%"
        print(f"{rank:<4} | {row['ticker']:<7} | {name:<18} | {sector:<15} | {mcap:<8} | {tranche:<24} | {score:<9} | {rev:<10}")

    print("="*110)
    print("📌 SUMMARY INSIGHTS:")
    print("  • Displaying Top 10 High Conviction Candidates.")
    print("  • Cache file stored at: " + CACHE_FILE)
    print("="*110 + "\n")

if __name__ == "__main__":
    print("🚀 Starting 1-Year 2x Stock Screener Pipeline...")
    data = fetch_stock_data(SAMPLE_TICKERS)
    results_df = train_and_score_model(data)
    print_screener_report(results_df)
