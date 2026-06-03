import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger("QuantEdgeForecaster")

class QuantEdgeForecaster:
    def __init__(self):
        # Check if scikit-learn is available for regression forecasting
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            self.has_sklearn = True
            self.LinearRegression = LinearRegression
            self.RandomForestRegressor = RandomForestRegressor
            logger.info("Scikit-learn forecasters are available.")
        except ImportError:
            self.has_sklearn = False
            logger.warning("Scikit-learn not found. Falling back to Moving Average/Exponential Smoothing.")

    def fetch_historical_prices(self, ticker, days_back=120):
        """Fetches historical price data from yfinance for a single ticker."""
        import yfinance as yf
        logger.info(f"Fetching historical prices for {ticker} for the last {days_back} days...")
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns.values]
            df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            return df[["date", "close", "volume"]]
        except Exception as e:
            logger.error(f"Error fetching historical prices for forecasting {ticker}: {e}")
            return pd.DataFrame()

    def forecast_moving_average(self, prices, forecast_days=7, window=10):
        """Forecasts using simple moving average extension."""
        forecast = list(prices)
        for _ in range(forecast_days):
            next_val = np.mean(forecast[-window:])
            forecast.append(next_val)
        return forecast[-forecast_days:]

    def forecast_exponential_smoothing(self, prices, forecast_days=7, alpha=0.3):
        """Forecasts using single exponential smoothing."""
        # Calculate level
        level = prices[0]
        levels = [level]
        for val in prices[1:]:
            level = alpha * val + (1 - alpha) * level
            levels.append(level)
        
        # Forecast is simply the last level extended
        last_level = levels[-1]
        return [last_level] * forecast_days

    def forecast_linear_regression(self, prices, dates, forecast_days=7):
        """Forecasts using a scikit-learn LinearRegression model with lag features."""
        if not self.has_sklearn:
            return self.forecast_exponential_smoothing(prices, forecast_days)

        # Create DataFrame with lag features
        df = pd.DataFrame({"close": prices})
        df["time"] = np.arange(len(df))
        
        # Add lag features
        for lag in range(1, 6):
            df[f"lag_{lag}"] = df["close"].shift(lag)
            
        df = df.dropna().reset_index(drop=True)
        
        if df.empty:
            return self.forecast_exponential_smoothing(prices, forecast_days)
            
        # Features & target
        features = ["time"] + [f"lag_{lag}" for lag in range(1, 6)]
        X = df[features].values
        y = df["close"].values
        
        # Train model
        model = self.LinearRegression()
        model.fit(X, y)
        
        # Forecast autoregressively
        forecast = []
        current_lags = list(prices[-5:])
        last_time = df["time"].iloc[-1]
        
        for i in range(forecast_days):
            next_time = last_time + 1 + i
            # Prepare feature vector: time index and lags in reverse order (t-1, t-2, t-3, t-4, t-5)
            x_input = np.array([[next_time] + current_lags[-5:]])
            pred = float(model.predict(x_input)[0])
            forecast.append(pred)
            current_lags.append(pred)
            
        return forecast

    def generate_forecast(self, ticker, method="regression", forecast_days=7):
        """Generates a 7-day price forecast and returns dates and predicted values."""
        df = self.fetch_historical_prices(ticker)
        if df.empty or len(df) < 15:
            logger.warning(f"Insufficient history to forecast {ticker}. Generating dummy/flat forecast.")
            # Flat forecast based on last price or a default
            last_price = 150.0
            if not df.empty:
                last_price = float(df["close"].iloc[-1])
            last_date = datetime.now()
            predictions = [last_price * (1 + 0.001 * i) for i in range(1, forecast_days + 1)]
            forecast_dates = [(last_date + timedelta(days=i)).date() for i in range(1, forecast_days + 1)]
        else:
            prices = df["close"].tolist()
            dates = df["date"].tolist()
            last_date = dates[-1]
            
            # Forecast prices
            if method == "regression" and self.has_sklearn:
                predictions = self.forecast_linear_regression(prices, dates, forecast_days)
            elif method == "exponential_smoothing":
                predictions = self.forecast_exponential_smoothing(prices, forecast_days)
            else:
                predictions = self.forecast_moving_average(prices, forecast_days)
                
            # Forecast dates (next business days or standard calendar days)
            forecast_dates = []
            curr_date = last_date
            while len(forecast_dates) < forecast_days:
                curr_date += timedelta(days=1)
                # If we want trading days only, we could filter out weekends.
                # Let's include weekends for a continuous calendar line, or business days.
                # Let's do business days (Monday-Friday) to match trading dates.
                if curr_date.weekday() < 5:
                    forecast_dates.append(curr_date.date())
                    
        results = []
        for d, p in zip(forecast_dates, predictions):
            results.append({
                "date": d.isoformat() if isinstance(d, (date, datetime)) else str(d),
                "predicted_price": round(float(p), 2)
            })
            
        return results

if __name__ == "__main__":
    forecaster = QuantEdgeForecaster()
    # Test
    res = forecaster.generate_forecast("AAPL", method="regression")
    print(res)
