import pandas as pd
from sqlalchemy import text
from datetime import date
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import engine
from config.settings import INDEX_BASE_VALUE

logger = logging.getLogger(__name__)

def calculate_adjusted_divisor(new_mcap, last_index_value):
    """
    Calculates a new divisor during rebalancing to ensure index continuity.
    Formula: New Divisor = New Market Cap / Last Index Value
    """
    if last_index_value == 0:
        return 0
    return new_mcap / last_index_value

def calculate_daily_index(start_date: date, end_date: date):
    """
    Calculates the daily index value between start_date and end_date.
    Manages the index divisor across rebalances using MSCI-standard continuity rules.
    """
    logger.info(f"Starting daily index calculation for period: {start_date} -> {end_date}")
    
    # Get all trading days in the range
    query_dates = text("""
        SELECT DISTINCT date FROM daily_prices 
        WHERE date >= :s_date AND date <= :e_date
        ORDER BY date
    """)
    dates_df = pd.read_sql(query_dates, engine, params={"s_date": start_date, "e_date": end_date})
    
    if dates_df.empty:
        logger.warning("No trading days identified in the requested range.")
        return
        
    trading_days = dates_df['date'].tolist()
    
    current_divisor = None
    last_index_value = None
    last_rebalance_date = None
    
    index_values_records = []
    
    for t_date in trading_days:
        # Find active constituents for this day
        query_const = text("""
            SELECT ticker, shares_in_index, rebalance_date 
            FROM index_constituents
            WHERE rebalance_date = (
                SELECT MAX(rebalance_date) 
                FROM index_constituents 
                WHERE rebalance_date <= :t_date
            )
        """)
        const_df = pd.read_sql(query_const, engine, params={"t_date": t_date})
        
        if const_df.empty:
            continue
            
        current_rebalance_date = const_df['rebalance_date'].iloc[0]
        
        # Get prices for these constituents on t_date
        tickers = list(const_df['ticker'].tolist())
        # SQLite doesn't support 'ticker IN :tickers' with a tuple in raw text() easily.
        # We'll use a safer formatted string approach for the IN clause.
        placeholders = ', '.join([':t' + str(i) for i in range(len(tickers))])
        params = {"t_date": t_date}
        for i, t in enumerate(tickers):
            params['t' + str(i)] = t
            
        query_prices = text(f"""
            SELECT ticker, close_price 
            FROM daily_prices 
            WHERE date = :t_date AND ticker IN ({placeholders})
        """)
        prices_df = pd.read_sql(query_prices, engine, params=params)
        
        # Merge prices with constituents
        merged = pd.merge(const_df, prices_df, on='ticker', how='inner')
        if merged.empty:
            continue
            
        current_mcap = (merged['shares_in_index'] * merged['close_price']).sum()
        
        # Divisor Logic
        if current_divisor is None:
            # Day 1 initialization
            current_divisor = current_mcap / INDEX_BASE_VALUE
            index_value = INDEX_BASE_VALUE
        elif current_rebalance_date != last_rebalance_date:
            # Rebalance Event: Adjust divisor to prevent artificial jump
            current_divisor = calculate_adjusted_divisor(current_mcap, last_index_value)
            index_value = current_mcap / current_divisor
        else:
            # Standard trading day
            index_value = current_mcap / current_divisor
            
        last_index_value = index_value
        last_rebalance_date = current_rebalance_date
        
        index_values_records.append({
            'date': t_date,
            'value': index_value,
            'market_cap': current_mcap,
            'divisor': current_divisor
        })
        
    if index_values_records:
        iv_df = pd.DataFrame(index_values_records)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM index_values WHERE date >= :s AND date <= :e"), {"s": start_date, "e": end_date})
        iv_df.to_sql('index_values', engine, if_exists='append', index=False)
        logger.info(f"Daily calculation complete. {len(iv_df)} index points recorded.")
