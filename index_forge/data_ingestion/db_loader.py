import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_db, engine
from db.models import Equity, DailyPrice

import logging

logger = logging.getLogger(__name__)

def load_equities_to_db(df: pd.DataFrame):
    """Loads equity metadata to db."""
    logger.info(f"Loading {len(df)} equities to database...")
    df.to_sql('equities', con=engine, if_exists='append', index=False)
    logger.info("Equities loaded.")

def load_prices_to_db(df: pd.DataFrame):
    """Loads daily prices to db. Handles massive datasets."""
    logger.info(f"Loading {len(df)} daily price records to database...")
    # calculating ADTV (Average Daily Traded Volume) - 20 day smoothing
    df = df.sort_values(by=['ticker', 'date'])
    df['adtv_20d'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['adtv_20d'] = df['adtv_20d'].fillna(0)
    
    # We write chunks to avoid memory overflow on huge tables
    df.to_sql('daily_prices', con=engine, if_exists='append', index=False, chunksize=10000)
    logger.info("Prices loaded.")

def clear_database():
    """Wipes all data for a fresh ingestion run."""
    logger.info("Clearing existing database records for fresh ingestion...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM index_values;"))
        conn.execute(text("DELETE FROM index_constituents;"))
        conn.execute(text("DELETE FROM daily_prices;"))
        conn.execute(text("DELETE FROM equities;"))
