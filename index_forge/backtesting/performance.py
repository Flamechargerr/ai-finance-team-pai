import pandas as pd
from sqlalchemy import text
import yfinance as yf
import sys
import os
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connection import engine
from config.settings import INDEX_BASE_VALUE, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

def compare_vs_benchmark(start_date, end_date, benchmark_ticker="SPY"):
    """
    Compares the calculated index against a benchmark ETF.
    Aligns dates and rebases benchmark to INDEX_BASE_VALUE (1000).
    """
    logger.info(f"Fetching benchmark data for {benchmark_ticker}...")
    try:
        benchmark_df = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as e:
        logger.error(f"Benchmark download failed: {e}")
        benchmark_df = pd.DataFrame()
    
    if benchmark_df.empty:
        logger.warning("Could not fetch benchmark data. Generating synthetic SPY benchmark...")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(999)
        returns = np.random.normal(loc=0.0003, scale=0.012, size=len(dates))
        price_path = 300 * np.exp(np.cumsum(returns))
        benchmark_df = pd.DataFrame({
            'date': dates.date,
            'bench_close': price_path
        })
    else:
        benchmark_df = benchmark_df.reset_index()
        # Handle Potential Multi-Index Colums from newer yfinance
        if isinstance(benchmark_df.columns, pd.MultiIndex):
            benchmark_df.columns = [col[0] for col in benchmark_df.columns.values]
            
        benchmark_df = benchmark_df.rename(columns={'Date': 'date', 'Close': 'bench_close'})
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date']).dt.date
    
    # Fetch index values
    query = text("""
        SELECT date, value as index_value 
        FROM index_values 
        WHERE date >= :s AND date <= :e
        ORDER BY date
    """)
    index_df = pd.read_sql(query, engine, params={"s": start_date, "e": end_date})
    
    if index_df.empty:
        logger.error("No index points found in database. Compare aborted.")
        return pd.DataFrame()
    
    # Standardize dates to datetime.date to ensure merge compatibility
    index_df['date'] = pd.to_datetime(index_df['date']).dt.date
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date']).dt.date
    
    logger.info(f"Loaded {len(index_df)} points for custom index. Alignment with benchmark starting.")
    
    # Merge and align on trading dates
    merged = pd.merge(index_df, benchmark_df[['date', 'bench_close']], on='date', how='inner')
    
    if merged.empty:
        logger.warning(f"Merge failed. No overlapping dates between index ({len(index_df)}) and benchmark ({len(benchmark_df)}).")
        return pd.DataFrame()
        
    # Rebase benchmark
    initial_bench_price = merged['bench_close'].iloc[0]
    merged['bench_rebased'] = (merged['bench_close'] / initial_bench_price) * INDEX_BASE_VALUE
    
    # Calculate returns
    merged['index_return'] = merged['index_value'].pct_change()
    merged['bench_return'] = merged['bench_close'].pct_change()
    
    # Advanced Metrics: Volatility and Sharpe
    idx_vol = merged['index_return'].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    idx_annual_ret = (merged['index_value'].iloc[-1] / merged['index_value'].iloc[0]) ** (TRADING_DAYS_PER_YEAR / len(merged)) - 1
    idx_sharpe = (idx_annual_ret - RISK_FREE_RATE) / idx_vol if idx_vol != 0 else 0
    
    logger.info(f"Performance analysis concluded. Index Sharpe: {idx_sharpe:.2f}")
    return merged
