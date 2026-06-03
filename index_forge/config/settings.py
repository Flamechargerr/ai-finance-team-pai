import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./indexforge.db")

# --- Index Calculation Settings ---
INDEX_BASE_VALUE = 1000.0
TARGET_CONSTITUENTS = 50
BUFFER_ZONE = 5  # Number of ranks for buffer-rule retention

# --- Data Ingestion Settings ---
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'

# We use a hardcoded mega-cap list for demonstration stability
DEMO_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
    "BRK-B", "UNH", "JNJ", "JPM", "XOM", "V", "PG", "HD", "MA",
    "CVX", "ABBV", "MRK", "PEP", "KO", "AVGO", "BAC", "PFE", "COST",
    "TMO", "CSCO", "MCD", "NKE", "ABT", "DHR", "DIS", "LIN", "CRM",
    "TXN", "WFC", "ADBE", "UPS", "PM", "MS", "VZ", "RTX", "HON",
    "INTC", "BMY", "NEE", "AMGN", "LOW", "CAT"
]

# --- Rebalancing Dates ---
# MSCI semi-annual schedule proxies: May and November ends
def get_rebalance_dates(start_year, end_year):
    dates_list = []
    for y in range(start_year, end_year + 1):
        dates_list.append(date(y, 5, 31))
        dates_list.append(date(y, 11, 30))
    return dates_list

# --- Risk Analysis ---
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate for Sharpe Ratio
TRADING_DAYS_PER_YEAR = 252
