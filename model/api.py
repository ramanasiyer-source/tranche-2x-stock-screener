import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Allow importing from model config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CURRENT_SCORES_FILE, PRICE_CACHE_DIR

# Setup FastAPI app
app = FastAPI(title="Tranche 2x Quant API")

# Add CORS to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SANDBOX_FILE = os.path.join(os.path.dirname(CURRENT_SCORES_FILE), 'sandbox_portfolio.json')


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


def _get_price_on_date(hist: pd.DataFrame, date: datetime) -> float | None:
    """Gets the closing price on or just before the specified date."""
    if hist is None or hist.empty:
        return None
    sub = hist[hist.index <= date]
    if sub.empty:
        return None
    close_col = 'Close' if 'Close' in sub.columns else sub.columns[3]
    return float(sub[close_col].iloc[-1])


def compute_performance_metrics(ticker: str, current_price: float) -> dict:
    hist = _load_price_hist(ticker)
    if hist is None or hist.empty:
        return {"wow": None, "mom": None, "qoq": None, "ytd": None, "high_52w": None, "low_52w": None}

    now = datetime.now()
    
    # Prices
    p_1y = _get_price_on_date(hist, now - timedelta(days=365))
    p_2y = _get_price_on_date(hist, now - timedelta(days=730))
    p_3y = _get_price_on_date(hist, now - timedelta(days=1095))
    
    # YTD price (last trading day of previous year)
    ytd_start_date = datetime(now.year - 1, 12, 31)
    p_ytd = _get_price_on_date(hist, ytd_start_date)

    def pct_change(p_old):
        if p_old and p_old > 0:
            return round((current_price / p_old - 1) * 100, 2)
        return None

    # 52w High/Low
    window_start = now - timedelta(days=365)
    window_hist = hist[hist.index >= window_start]
    close_col = 'Close' if 'Close' in hist.columns else hist.columns[3]
    high_52w = float(window_hist[close_col].max()) if not window_hist.empty else None
    low_52w = float(window_hist[close_col].min()) if not window_hist.empty else None

    return {
        "1y": pct_change(p_1y),
        "2y": pct_change(p_2y),
        "3y": pct_change(p_3y),
        "ytd": pct_change(p_ytd),
        "high_52w": round(high_52w, 2) if high_52w else None,
        "low_52w": round(low_52w, 2) if low_52w else None
    }


def _generate_nlp_reasoning(stock: dict) -> str:
    """Generate a human-readable explanation based on SHAP drivers."""
    drivers = stock.get("top_drivers", [])
    if not drivers:
        return "No SHAP feature drivers available for this stock."
    
    # Get top positive and top negative driver
    pos_drivers = [d for d in drivers if d["shap"] > 0]
    neg_drivers = [d for d in drivers if d["shap"] < 0]
    
    reason = f"The model gives {stock['ticker']} a {stock['doubling_prob_12m']*100:.1f}% chance to double over the next 12 months."
    
    if pos_drivers:
        top_pos = pos_drivers[0]
        reason += f" This is driven primarily by its strong {top_pos['feature'].replace('_', ' ')}."
        if len(pos_drivers) > 1:
            reason += f" Additionally, {pos_drivers[1]['feature'].replace('_', ' ')} contributed positively."
            
    if neg_drivers:
        top_neg = neg_drivers[0]
        reason += f" However, its {top_neg['feature'].replace('_', ' ')} is holding its score back."
        
    return reason


@app.get("/api/leaderboard")
def get_leaderboard():
    if not os.path.exists(CURRENT_SCORES_FILE):
        raise HTTPException(status_code=404, detail="Leaderboard not found. Run pipeline first.")
        
    with open(CURRENT_SCORES_FILE) as f:
        scores = json.load(f)
        
    # Process top 50 to save time
    results = []
    for s in scores[:50]:
        metrics = compute_performance_metrics(s["ticker"], s["current_price"])
        s["performance"] = metrics
        s["nlp_reasoning"] = _generate_nlp_reasoning(s)
        results.append(s)
        
    return {"leaderboard": results}


@app.get("/api/historical-price")
def get_historical_price(ticker: str, date: str):
    """Fetch the closing price for a specific date (YYYY-MM-DD)."""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
    hist = _load_price_hist(ticker)
    if hist is None:
        raise HTTPException(status_code=404, detail=f"Price history not found for {ticker}")
        
    price = _get_price_on_date(hist, dt)
    if price is None:
        raise HTTPException(status_code=404, detail=f"No price found for {ticker} on {date}")
        
    return {"ticker": ticker, "date": date, "price": round(price, 4)}


@app.get("/api/sandbox")
def get_sandbox():
    if os.path.exists(SANDBOX_FILE):
        with open(SANDBOX_FILE) as f:
            return json.load(f)
    return {"positions": []}


@app.post("/api/sandbox/add")
def add_to_sandbox(position: dict):
    # Expected: {"ticker": "XYZ", "quantity": 100, "buy_date": "YYYY-MM-DD", "buy_price": 50.0}
    sandbox = get_sandbox()
    
    sandbox["positions"].append(position)
    
    with open(SANDBOX_FILE, 'w') as f:
        json.dump(sandbox, f, indent=2)
        
    return {"status": "success", "sandbox": sandbox}

from fastapi.staticfiles import StaticFiles

# ...

# Serve static files for the UI
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
