import yfinance as yf
import pandas as pd
import logging
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DEMO_TICKERS

logger = logging.getLogger(__name__)

def get_universe_tickers():
    """Returns a defined mega-cap universe for demonstration stability."""
    return DEMO_TICKERS

def fetch_equity_metadata(tickers):
    """Generates metadata for assigned tickers."""
    metadata = []
    logger.info(f"Preparing metadata for {len(tickers)} constituents...")
    
    sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Defensive', 'Energy', 'Communication Services']
    for i, ticker_str in enumerate(tqdm(tickers, desc="Metadata Generation")):
        # Deterministic dummy data for demo to avoid yfinance rate limits on 'info'
        metadata.append({
            'ticker': ticker_str,
            'company_name': f"{ticker_str} Corp",
            'sector': sectors[i % len(sectors)],
            'industry': 'Diversified',
            'shares_outstanding': 1000000000 + (hash(ticker_str) % 5000000000)
        })
            
    return pd.DataFrame(metadata)

def fetch_historical_prices(tickers, start_date, end_date):
    """Fetches historical price data via yfinance, with synthetic fallback for stability."""
    logger.info(f"Downloading historical data for {len(tickers)} symbols...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False, threads=True, progress=False)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        data = pd.DataFrame()

    records = []
    if not data.empty:
        for ticker in tickers:
            if ticker in data:
                df = data[ticker].copy().dropna(subset=['Close'])
                if not df.empty:
                    df = df.reset_index()
                    df['ticker'] = ticker
                    df = df.rename(columns={'Date': 'date', 'Close': 'close_price', 'Adj Close': 'adj_close', 'Volume': 'volume'})
                    records.extend(df[['ticker', 'date', 'close_price', 'adj_close', 'volume']].to_dict('records'))

    flat_df = pd.DataFrame(records)

    # Standard Fallback: If network issues block Yahoo Finance, generate synthetic market data
    if flat_df.empty:
        logger.warning("No data retrieved from API. Switching to synthetic market generation...")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        synthetic_records = []
        for ticker in tickers:
            np.random.seed(hash(ticker) % (2**32))
            start_p = np.random.uniform(50, 500)
            returns = np.random.normal(loc=0.0003, scale=0.015, size=len(dates))
            price_path = start_p * np.exp(np.cumsum(returns))
            for i, d in enumerate(dates):
                synthetic_records.append({
                    'ticker': ticker,
                    'date': d.date(),
                    'close_price': float(price_path[i]),
                    'adj_close': float(price_path[i]),
                    'volume': int(np.random.uniform(1000000, 10000000))
                })
        flat_df = pd.DataFrame(synthetic_records)
    else:
        flat_df['date'] = pd.to_datetime(flat_df['date']).dt.date
        logger.info(f"Successfully ingested {len(flat_df)} price records.")
        
    return flat_df
