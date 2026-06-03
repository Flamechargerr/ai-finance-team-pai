import pandas as pd
from sqlalchemy import text
from datetime import date
import sys
import os
import numpy as np
import logging
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import engine
from config.settings import TARGET_CONSTITUENTS, BUFFER_ZONE

logger = logging.getLogger(__name__)

# Mock free-float factor generator for demonstration (real MSCI uses proprietary research)
def get_mock_free_float(ticker):
    seed = int(hashlib.sha256(ticker.encode("utf-8")).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    return np.random.uniform(0.70, 1.0)

def calculate_rebalance(rebalance_date: date):
    """
    Applies rules to select index constituents for a given date.
    Implements MSCI-style buffer rules for retention.
    """
    logger.info(f"Executing rebalance evaluation for {rebalance_date}...")
    
    # 1. Fetch universe state as of rebalance_date
    query = text("""
        SELECT e.ticker, e.shares_outstanding, p.close_price, p.adtv_20d
        FROM equities e
        JOIN daily_prices p ON e.ticker = p.ticker
        WHERE p.date = :r_date
    """)
    df = pd.read_sql(query, engine, params={"r_date": rebalance_date})
    
    if df.empty:
        logger.warning(f"No price data found for {rebalance_date}. Rebalance cannot proceed.")
        return pd.DataFrame()
        
    # 2. Liquidity Screen (e.g., ADTV > 500,000)
    df = df[df['adtv_20d'] > 500000].copy()
    
    # 3. Calculate free-float adjusted market cap
    df['free_float_factor'] = df['ticker'].apply(get_mock_free_float)
    df['market_cap'] = df['shares_outstanding'] * df['close_price']
    df['ff_market_cap'] = df['market_cap'] * df['free_float_factor']
    
    # 4. Rank by ff_market_cap
    df['rank'] = df['ff_market_cap'].rank(ascending=False)
    
    # 5. Buffer rules
    curr_query = text("""
        SELECT ticker FROM index_constituents 
        WHERE rebalance_date = (SELECT MAX(rebalance_date) FROM index_constituents WHERE rebalance_date < :r_date)
    """)
    current_tickers = set(pd.read_sql(curr_query, engine, params={"r_date": rebalance_date})['ticker'])
    
    selected_tickers = set()
    
    # Rule A: Select top (Target - Buffer) unconditionally (automatic inclusion)
    automatic_inclusion = set(df[df['rank'] <= (TARGET_CONSTITUENTS - BUFFER_ZONE)]['ticker'])
    selected_tickers.update(automatic_inclusion)
    
    # Rule B: Existing constituents between (Target - Buffer) and (Target + Buffer) are retained
    buffer_range = df[(df['rank'] > (TARGET_CONSTITUENTS - BUFFER_ZONE)) & (df['rank'] <= (TARGET_CONSTITUENTS + BUFFER_ZONE))]
    retained = set(buffer_range['ticker']).intersection(current_tickers)
    selected_tickers.update(retained)
    
    # Rule C: Fill remaining slots with the highest ranked newcomers
    needed = TARGET_CONSTITUENTS - len(selected_tickers)
    if needed > 0:
        candidates = df[~df['ticker'].isin(selected_tickers)].sort_values(by='rank')
        selected_tickers.update(set(candidates.head(needed)['ticker']))
        
    final_df = df[df['ticker'].isin(selected_tickers)].copy()
    
    # Weights calculation
    total_ff_mcap = final_df['ff_market_cap'].sum()
    final_df['weight'] = final_df['ff_market_cap'] / total_ff_mcap
    final_df['shares_in_index'] = final_df['shares_outstanding'] * final_df['free_float_factor']
    final_df['rebalance_date'] = rebalance_date
    
    logger.info(f"Rebalance finalized: {len(final_df)} constituents selected.")
    return final_df[['ticker', 'rebalance_date', 'weight', 'shares_in_index', 'free_float_factor']]

def apply_rebalance(rebalance_date: date):
    index_df = calculate_rebalance(rebalance_date)
    if not index_df.empty:
        index_df.to_sql('index_constituents', engine, if_exists='append', index=False)
