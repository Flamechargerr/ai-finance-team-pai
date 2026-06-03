import pandas as pd
from sqlalchemy import text
from db.connection import engine
import logging

logger = logging.getLogger(__name__)

def calculate_turnover():
    """
    Calculates portfolio turnover at each rebalance date.
    Turnover = sum(abs(New Weight_i - Old Weight_i)) / 2
    """
    logger.info("Calculating constituent turnover attribution...")
    query = text("""
        SELECT rebalance_date, ticker, weight 
        FROM index_constituents
        ORDER BY rebalance_date, ticker
    """)
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return pd.DataFrame()
        
    dates = df['rebalance_date'].unique()
    turnovers = []
    
    for i in range(1, len(dates)):
        old_date = dates[i-1]
        new_date = dates[i]
        
        old_weights = df[df['rebalance_date'] == old_date].set_index('ticker')['weight']
        new_weights = df[df['rebalance_date'] == new_date].set_index('ticker')['weight']
        
        # Align indexes
        all_tickers = list(set(old_weights.index).union(set(new_weights.index)))
        old_w = old_weights.reindex(all_tickers).fillna(0)
        new_w = new_weights.reindex(all_tickers).fillna(0)
        
        turnover = (abs(new_w - old_w).sum()) / 2.0
        turnovers.append({'rebalance_date': new_date, 'turnover_pct': turnover * 100})
        
    return pd.DataFrame(turnovers)

def calculate_sector_drift():
    """
    Groups constituents by sector to track allocation drift over time.
    """
    query = text("""
        SELECT i.rebalance_date, e.sector, SUM(i.weight) as sector_weight
        FROM index_constituents i
        JOIN equities e ON i.ticker = e.ticker
        GROUP BY i.rebalance_date, e.sector
        ORDER BY i.rebalance_date, sector_weight DESC
    """)
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return pd.DataFrame()
        
    # Pivot for report readability
    pivot_df = df.pivot(index='rebalance_date', columns='sector', values='sector_weight').fillna(0) * 100
    return pivot_df
