import os
import time
import logging
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# Import database connection and schemas
from core.database import get_session, get_engine, init_db, EquitySignal, ETLRun

logger = logging.getLogger("QuantEdgeETL")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# A comprehensive list of 50+ equities for the watchlist
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JNJ", "V",
    "PG", "JPM", "UNH", "MA", "HD", "DIS", "NFLX", "ADBE", "CSCO", "PEP",
    "INTC", "T", "PFE", "MRK", "WMT", "KO", "XOM", "CVX", "ABBV", "BAC",
    "COST", "AMD", "QCOM", "TXN", "HON", "LMT", "NEE", "LIN", "PM", "SBUX",
    "CVS", "MDT", "ORCL", "IBM", "CRM", "NKE", "UNP", "GS", "MS", "CAT",
    "HON", "GE", "TXN", "AMGN", "DE"
]
# Remove duplicates
DEFAULT_WATCHLIST = sorted(list(set(DEFAULT_WATCHLIST)))

class QuantEdgeETL:
    def __init__(self, use_pyspark=True):
        self.use_pyspark = use_pyspark
        self.spark = None
        if self.use_pyspark:
            try:
                from pyspark.sql import SparkSession
                # Setup a local PySpark session
                self.spark = (
                    SparkSession.builder
                    .appName("QuantEdgeETL")
                    .master("local[*]")
                    .config("spark.sql.shuffle.partitions", "4")
                    .config("spark.driver.memory", "2g")
                    .getOrCreate()
                )
                logger.info("Successfully initialized PySpark ETL Session.")
            except Exception as e:
                logger.warning(f"Could not initialize PySpark: {e}. Falling back to Pandas ETL.")
                self.use_pyspark = False
                self.spark = None

    def fetch_historical_data(self, tickers, days_back=365):
        """Downloads historical price data from yfinance and returns a consolidated Pandas DataFrame."""
        import yfinance as yf
        logger.info(f"Fetching {days_back} days of historical data for {len(tickers)} tickers...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    continue
                df = df.reset_index()
                # Handle MultiIndex column names from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns.values]
                
                df = df.rename(columns={
                    "Date": "date",
                    "Close": "close",
                    "Volume": "volume",
                    "Open": "open",
                    "High": "high",
                    "Low": "low"
                })
                df["ticker"] = ticker
                # Clean dates
                df["date"] = pd.to_datetime(df["date"]).dt.date
                all_data.append(df[["ticker", "date", "close", "volume"]])
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

    def run_pandas_etl(self, pdf):
        """Calculates indicators and signals using Pandas."""
        logger.info("Running Pandas-based ETL calculations...")
        results = []
        
        # Group by ticker and sort by date
        pdf = pdf.sort_values(["ticker", "date"])
        
        for ticker, group in pdf.groupby("ticker"):
            group = group.copy().reset_index(drop=True)
            if len(group) < 14:  # Need at least 14 days for RSI
                continue
                
            # Moving Averages
            group["sma_50"] = group["close"].rolling(window=50, min_periods=1).mean()
            group["sma_200"] = group["close"].rolling(window=200, min_periods=1).mean()
            
            # Volume 20d Average
            group["vol_avg_20d"] = group["volume"].rolling(window=20, min_periods=1).mean()
            
            # RSI 14d calculation
            delta = group["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            
            # Wilder's smoothing or simple rolling mean fallback
            # We'll use rolling mean for simplicity and robustness
            rs = avg_gain / (avg_loss + 1e-9)
            group["rsi"] = 100 - (100 / (1 + rs))
            
            # Generate Signals
            for i in range(len(group)):
                row = group.iloc[i]
                ticker_val = row["ticker"]
                date_val = row["date"]
                close_val = float(row["close"])
                volume_val = int(row["volume"])
                
                sma50 = row["sma_50"]
                sma200 = row["sma_200"]
                rsi = row["rsi"]
                vol_avg = row["vol_avg_20d"]
                
                # Check for signal logic
                if pd.isna(sma50) or pd.isna(sma200) or pd.isna(rsi):
                    # Skip rows that don't have all indicators computed
                    continue
                    
                # Signal Rules
                # Golden Cross: 50 SMA > 200 SMA is bullish. Death Cross: 50 SMA < 200 SMA is bearish.
                # Oversold RSI (<30) is bullish. Overbought RSI (>70) is bearish.
                # Volume confirmation: volume > 1.2 * 20d average volume increases confidence.
                
                is_golden_cross = sma50 > sma200
                is_oversold = rsi < 30
                is_overbought = rsi > 70
                high_volume = volume_val > (1.2 * vol_avg) if not pd.isna(vol_avg) and vol_avg > 0 else False
                
                signal_str = "HOLD"
                confidence = 0.5
                
                if (is_golden_cross and not is_overbought) or is_oversold:
                    signal_str = "BUY"
                    # Determine confidence score
                    base_conf = 0.6
                    if is_oversold:
                        base_conf += 0.15
                    if is_golden_cross:
                        base_conf += 0.10
                    if high_volume:
                        base_conf += 0.10
                    confidence = min(0.98, base_conf)
                elif (not is_golden_cross and not is_oversold) or is_overbought:
                    signal_str = "SELL"
                    base_conf = 0.6
                    if is_overbought:
                        base_conf += 0.15
                    if not is_golden_cross:
                        base_conf += 0.10
                    if high_volume:
                        base_conf += 0.10
                    confidence = min(0.98, base_conf)
                else:
                    signal_str = "HOLD"
                    confidence = 0.5
                    
                results.append({
                    "ticker": ticker_val,
                    "date": date_val,
                    "signal": signal_str,
                    "confidence": float(confidence),
                    "price": close_val,
                    "volume": volume_val
                })
                
        return pd.DataFrame(results)

    def run_spark_etl(self, pdf):
        """Calculates indicators and signals using PySpark."""
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        logger.info("Running PySpark-based ETL calculations...")
        # Load local PDF into Spark DataFrame
        # Standardize schema
        df = self.spark.createDataFrame(pdf)
        
        # Define Windows
        # 50d and 200d average windows
        win_50 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-49, 0)
        win_200 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-199, 0)
        win_20 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-19, 0)
        
        # Calculate Moving Averages and Volume Averages
        df_ind = df.withColumn("sma_50", F.avg("close").over(win_50))
        df_ind = df_ind.withColumn("sma_200", F.avg("close").over(win_200))
        df_ind = df_ind.withColumn("vol_avg_20d", F.avg("volume").over(win_20))
        
        # Calculate RSI
        # 1. Price difference from previous day
        win_lead = Window.partitionBy("ticker").orderBy("date")
        df_rsi = df_ind.withColumn("prev_close", F.lag("close", 1).over(win_lead))
        df_rsi = df_rsi.withColumn("diff", F.col("close") - F.col("prev_close"))
        
        # 2. Gain/Loss
        df_rsi = df_rsi.withColumn("gain", F.when(F.col("diff") > 0, F.col("diff")).otherwise(0.0))
        df_rsi = df_rsi.withColumn("loss", F.when(F.col("diff") < 0, -F.col("diff")).otherwise(0.0))
        
        # 3. 14d Average Gain/Loss
        win_14 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-13, 0)
        df_rsi = df_rsi.withColumn("avg_gain", F.avg("gain").over(win_14))
        df_rsi = df_rsi.withColumn("avg_loss", F.avg("loss").over(win_14))
        
        # 4. RSI value
        df_rsi = df_rsi.withColumn("rs", F.col("avg_gain") / (F.col("avg_loss") + 1e-9))
        df_rsi = df_rsi.withColumn("rsi", 100.0 - (100.0 / (1.0 + F.col("rs"))))
        
        # Filter rows where indicators are properly populated (require at least 14 rows for RSI, but 50 for SMA_50, 200 for SMA_200)
        # To make sure we have data, we'll keep dates where we have enough lookback
        # But to be safe and get signals for recent dates, we'll filter rows where sma_50, sma_200, and rsi are NOT null
        df_valid = df_rsi.filter(df_rsi.sma_50.isNotNull() & df_rsi.sma_200.isNotNull() & df_rsi.rsi.isNotNull())
        
        # Calculate Buy/Hold/Sell signal and confidence in Spark SQL logic
        df_signals = df_valid.withColumn(
            "is_golden_cross", F.col("sma_50") > F.col("sma_200")
        ).withColumn(
            "is_oversold", F.col("rsi") < 30.0
        ).withColumn(
            "is_overbought", F.col("rsi") > 70.0
        ).withColumn(
            "high_volume", F.col("volume") > (1.2 * F.col("vol_avg_20d"))
        )
        
        # Buy/Sell signal expression
        df_signals = df_signals.withColumn(
            "signal",
            F.when((F.col("is_golden_cross") & ~F.col("is_overbought")) | F.col("is_oversold"), "BUY")
            .when((~F.col("is_golden_cross") & ~F.col("is_oversold")) | F.col("is_overbought"), "SELL")
            .otherwise("HOLD")
        )
        
        # Confidence score expression
        df_signals = df_signals.withColumn(
            "confidence",
            F.when(F.col("signal") == "BUY", 
                F.when(F.col("is_oversold") & F.col("is_golden_cross") & F.col("high_volume"), 0.95)
                .when(F.col("is_oversold") & F.col("is_golden_cross"), 0.85)
                .when(F.col("is_oversold") | F.col("is_golden_cross"), 0.70)
                .otherwise(0.60)
            ).when(F.col("signal") == "SELL",
                F.when(F.col("is_overbought") & ~F.col("is_golden_cross") & F.col("high_volume"), 0.95)
                .when(F.col("is_overbought") & ~F.col("is_golden_cross"), 0.85)
                .when(F.col("is_overbought") | ~F.col("is_golden_cross"), 0.70)
                .otherwise(0.60)
            ).otherwise(0.50)
        )
        
        # Select target columns
        df_final = df_signals.select("ticker", "date", "signal", "confidence", "close", "volume")
        
        # Convert back to Pandas for database loading
        res_pdf = df_final.toPandas()
        res_pdf = res_pdf.rename(columns={"close": "price"})
        return res_pdf

    def save_signals_to_db(self, signals_df):
        """Upserts signals into the database."""
        if signals_df.empty:
            logger.info("No signals generated to save to DB.")
            return 0
            
        session = get_session()
        records_saved = 0
        
        try:
            # We want to insert the latest signal per ticker-date combination.
            # To avoid unique constraint violations, we will insert or update.
            # SQLite / PG upsert can be done in raw SQL, but we'll do it via SQLAlchemy session.
            
            logger.info(f"Upserting {len(signals_df)} signals to the database...")
            
            # Since loading all at once might be slow, let's load in batches or process efficiently.
            # For simplicity, we filter to keep only the latest date's signals, or all.
            # Actually, the user wants a full history of signals, but if we rerun, we might overwrite.
            # We can upsert.
            
            # Let's get existing ticker-dates to decide insert/update
            # But query in chunks to avoid memory overflow for large tables
            existing_signals = {}
            # Check how many signals exist
            count = session.query(EquitySignal).count()
            if count < 100000:
                for sig in session.query(EquitySignal.ticker, EquitySignal.date, EquitySignal.id).all():
                    existing_signals[(sig.ticker, sig.date)] = sig.id

            batch = []
            for _, row in signals_df.iterrows():
                t_key = (row["ticker"], row["date"])
                
                # Check if signal is already in DB
                if t_key in existing_signals:
                    # Update
                    sig_id = existing_signals[t_key]
                    sig = session.query(EquitySignal).filter_by(id=sig_id).first()
                    if sig:
                        sig.signal = row["signal"]
                        sig.confidence = float(row["confidence"])
                        sig.price = float(row["price"])
                        sig.volume = int(row["volume"])
                else:
                    # Insert
                    new_sig = EquitySignal(
                        ticker=row["ticker"],
                        date=row["date"],
                        signal=row["signal"],
                        confidence=float(row["confidence"]),
                        price=float(row["price"]),
                        volume=int(row["volume"])
                    )
                    session.add(new_sig)
                
                records_saved += 1
                if len(session.new) >= 500 or len(session.dirty) >= 500:
                    session.commit()
                    
            session.commit()
            logger.info(f"Successfully saved {records_saved} signals to DB.")
            return records_saved
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving signals to DB: {e}")
            raise e
        finally:
            session.close()

    def execute_pipeline(self, tickers=None, days_back=365):
        """Runs the entire ETL pipeline and logs metrics in database."""
        start_time = time.time()
        status = "FAILED"
        records_processed = 0
        
        if tickers is None:
            tickers = DEFAULT_WATCHLIST
            
        try:
            # 1. Initialize tables if not already done
            init_db()
            
            # 2. Fetch prices
            pdf = self.fetch_historical_data(tickers, days_back)
            if pdf.empty:
                logger.warning("No historical price data fetched. ETL finished with 0 records.")
                status = "SUCCESS"
            else:
                # 3. Compute indicators and signals
                if self.use_pyspark and self.spark is not None:
                    signals_df = self.run_spark_etl(pdf)
                else:
                    signals_df = self.run_pandas_etl(pdf)
                
                # 4. Save to DB
                records_processed = self.save_signals_to_db(signals_df)
                status = "SUCCESS"
                
        except Exception as e:
            logger.error(f"ETL pipeline run failed: {e}")
            status = "FAILED"
            raise e
        finally:
            duration = time.time() - start_time
            # Record ETL run metrics
            session = get_session()
            try:
                run_log = ETLRun(
                    run_timestamp=datetime.utcnow(),
                    duration=float(duration),
                    records_processed=records_processed,
                    status=status
                )
                session.add(run_log)
                session.commit()
                logger.info(f"Recorded ETL run. Status: {status}, Processed: {records_processed}, Duration: {duration:.2f}s")
            except Exception as le:
                logger.error(f"Failed to record ETL run details to database: {le}")
            finally:
                session.close()
                
        return {
            "status": status,
            "records_processed": records_processed,
            "duration_seconds": duration
        }

if __name__ == "__main__":
    # Test script run
    etl = QuantEdgeETL(use_pyspark=False)
    # Run on a smaller watchlist for testing
    etl.execute_pipeline(tickers=["AAPL", "MSFT", "GOOGL"], days_back=100)
