import argparse
import sys
import os
from datetime import date
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.logging_config import setup_logging
from config.settings import START_DATE, END_DATE, DEMO_TICKERS, get_rebalance_dates
from db.init_db import init_db
from data_ingestion.yfinance_client import get_universe_tickers, fetch_equity_metadata, fetch_historical_prices
from data_ingestion.db_loader import load_equities_to_db, load_prices_to_db, clear_database
from index_math.rebalancing import apply_rebalance
from index_math.construction import calculate_daily_index
from backtesting.performance import compare_vs_benchmark
from backtesting.attribution import calculate_turnover, calculate_sector_drift
from backtesting.visualizer import plot_performance
from reports.generator import generate_markdown_report

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="IndexForge - Equity Index Construction Engine")
    parser.add_argument('--setup', action='store_true', help='Initialize database tables')
    parser.add_argument('--ingest', action='store_true', help='Download universe and pricing data')
    parser.add_argument('--run-engine', action='store_true', help='Run semi-annual rebalances and index calculations')
    parser.add_argument('--backtest', action='store_true', help='Generate attribution and performance reports')
    parser.add_argument('--visualize', action='store_true', help='Generate performance chart plot')
    parser.add_argument('--full-run', action='store_true', help='Execute all phases sequentially')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.setup or args.full_run:
        init_db()

    if args.ingest or args.full_run:
        logger.info("Starting automated data ingestion...")
        clear_database()
        tickers = get_universe_tickers()
        
        meta_df = fetch_equity_metadata(tickers)
        if not meta_df.empty:
            load_equities_to_db(meta_df)
            
        prices_df = fetch_historical_prices(tickers, start_date=START_DATE, end_date=END_DATE)
        if not prices_df.empty:
            load_prices_to_db(prices_df)

    if args.run_engine or args.full_run:
        logger.info("Targeting semi-annual rebalancing windows (MSCI Schedule)...")
        rebalance_dates = get_rebalance_dates(int(START_DATE[:4]), int(END_DATE[:4]))
        rebalance_dates = [d for d in rebalance_dates if pd.to_datetime(START_DATE).date() <= d <= pd.to_datetime(END_DATE).date()]
        
        for r_date in rebalance_dates:
            apply_rebalance(r_date)
            
        calculate_daily_index(pd.to_datetime(START_DATE).date(), pd.to_datetime(END_DATE).date())
        
    if args.backtest or args.full_run or args.visualize:
        logger.info("Executing performance and risk attribution backtest...")
        perf_df = compare_vs_benchmark(pd.to_datetime(START_DATE).date(), pd.to_datetime(END_DATE).date(), "SPY")
        
        if args.backtest or args.full_run:
            turnover_df = calculate_turnover()
            sector_df = calculate_sector_drift()
            generate_markdown_report(perf_df, turnover_df, sector_df, "indexforge_report.md")
            
        if args.visualize or args.full_run:
            plot_performance(perf_df)
            
        logger.info("Engineering cycle complete. Summary report generated.")

if __name__ == "__main__":
    main()
