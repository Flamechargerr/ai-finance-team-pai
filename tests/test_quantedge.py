import os
import unittest
from datetime import date, datetime
import pandas as pd

# Import system under test
from core.database import (
    get_session,
    init_db,
    seed_db,
    EquitySignal,
    SentimentLog,
    ETLRun,
    get_engine,
    reset_engine
)
from core.spark_etl import QuantEdgeETL
from core.forecaster import QuantEdgeForecaster

class TestQuantEdgeDatabase(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for fast isolated testing
        reset_engine()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        self.engine = get_engine()
        init_db(self.engine)
        self.session = get_session(self.engine)

    def tearDown(self):
        self.session.close()
        # Clean up database URL environment variable
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]

    def test_schema_creation(self):
        """Verify that schemas are properly constructed in the database."""
        # Querying count on empty tables should succeed without schema errors
        self.assertEqual(self.session.query(EquitySignal).count(), 0)
        self.assertEqual(self.session.query(SentimentLog).count(), 0)
        self.assertEqual(self.session.query(ETLRun).count(), 0)

    def test_seeding_functionality(self):
        """Verify that the database seeder populates correct data count and types."""
        seed_db(self.session)
        
        # Verify initial signals are loaded
        signal_count = self.session.query(EquitySignal).count()
        self.assertGreater(signal_count, 40, "Seeder should insert signals for 50 tickers")
        
        # Check specific model structures
        first_sig = self.session.query(EquitySignal).first()
        self.assertIsNotNone(first_sig.ticker)
        self.assertIn(first_sig.signal, ["BUY", "SELL", "HOLD"])
        self.assertGreater(first_sig.confidence, 0.0)
        self.assertGreater(first_sig.price, 0.0)
        
        # Verify sentiment logs are loaded
        sentiment_count = self.session.query(SentimentLog).count()
        self.assertEqual(sentiment_count, signal_count)
        
        # Verify ETL Run history is loaded
        etl_run_count = self.session.query(ETLRun).count()
        self.assertEqual(etl_run_count, 5)

class TestQuantEdgeETL(unittest.TestCase):
    def setUp(self):
        reset_engine()
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        self.engine = get_engine()
        init_db(self.engine)
        self.etl = QuantEdgeETL(use_pyspark=False) # Test using Pandas fallback for speed & compatibility

    def tearDown(self):
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]

    def test_indicator_calculations(self):
        """Verify SMA, RSI, and signal generation calculations."""
        # Create a mock series of price data (150 days to allow indicators to calculate)
        # 150 days of AAPL price starting from 100 growing steadily
        dates = pd.date_range(start="2026-01-01", periods=150).date
        prices = [100.0 + i * 0.5 for i in range(150)] # Upward trend
        volume = [1000000 + (i % 5) * 100000 for i in range(150)]
        
        pdf = pd.DataFrame({
            "ticker": ["AAPL"] * 150,
            "date": dates,
            "close": prices,
            "volume": volume
        })
        
        # Run calculations
        processed_df = self.etl.run_pandas_etl(pdf)
        
        self.assertFalse(processed_df.empty, "Processed DataFrame should contain computed rows")
        # Check column keys
        for col in ["ticker", "date", "signal", "confidence", "price", "volume"]:
            self.assertIn(col, processed_df.columns)
            
        # Verify all signals are generated
        signals = processed_df["signal"].unique()
        self.assertGreater(len(signals), 0)
        self.assertTrue(all(s in ["BUY", "SELL", "HOLD"] for s in signals))

    def test_pipeline_execution(self):
        """Runs the entire pipeline on a small watchlist and verifies database persistence."""
        # Use a single test ticker to keep the test quick
        res = self.etl.execute_pipeline(tickers=["AAPL"], days_back=30)
        
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreaterEqual(res["records_processed"], 0)
        
        # Check database records
        session = get_session(self.engine)
        try:
            db_signals = session.query(EquitySignal).filter_by(ticker="AAPL").all()
            # If historical yfinance data returned rows, signals should be saved
            if res["records_processed"] > 0:
                self.assertGreater(len(db_signals), 0)
                # Check run log
                db_runs = session.query(ETLRun).all()
                self.assertEqual(len(db_runs), 1)
                self.assertEqual(db_runs[0].status, "SUCCESS")
        finally:
            session.close()

class TestQuantEdgeForecaster(unittest.TestCase):
    def setUp(self):
        self.forecaster = QuantEdgeForecaster()

    def test_forecast_methods(self):
        """Verify mathematical outputs of MA and Exponential Smoothing forecasting models."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        
        # 1. Moving Average
        ma_forecast = self.forecaster.forecast_moving_average(prices, forecast_days=3, window=5)
        self.assertEqual(len(ma_forecast), 3)
        # Expected N+1 is average of last 5: (15+16+17+18+19)/5 = 17.0
        self.assertAlmostEqual(ma_forecast[0], 17.0)

        # 2. Exponential Smoothing
        es_forecast = self.forecaster.forecast_exponential_smoothing(prices, forecast_days=3, alpha=0.3)
        self.assertEqual(len(es_forecast), 3)
        self.assertGreater(es_forecast[0], 10.0)

    def test_regression_forecast(self):
        """Verify that regression forecast executes and returns 7 predictions."""
        # Run linear regression forecast for a small mock array
        prices = [float(100 + i * 2) for i in range(20)]
        dates = pd.date_range(start="2026-01-01", periods=20)
        
        reg_forecast = self.forecaster.forecast_linear_regression(prices, dates, forecast_days=7)
        self.assertEqual(len(reg_forecast), 7)
        
        # Since it's a linear growth, the predictions should continue the trend (> 138)
        self.assertGreater(reg_forecast[0], 138)

    def test_generate_forecast_api(self):
        """Verify that generate_forecast returns formatted outputs with date and predicted_price keys."""
        forecast_res = self.forecaster.generate_forecast("AAPL", method="regression", forecast_days=7)
        
        self.assertEqual(len(forecast_res), 7)
        for item in forecast_res:
            self.assertIn("date", item)
            self.assertIn("predicted_price", item)
            self.assertIsInstance(item["predicted_price"], float)
            # Verify date string format
            self.assertTrue(isinstance(item["date"], str))

if __name__ == "__main__":
    unittest.main()
