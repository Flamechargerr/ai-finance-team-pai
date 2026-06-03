import datetime as dt
import sys
import os

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_math import construction
from index_math.construction import calculate_adjusted_divisor

def test_divisor_adjustment_continuity():
    """
    Test that the divisor adjustment formula correctly maintains index value 
    given a change in market capitalization.
    Index_Value = Market_Cap / Divisor
    """
    last_index_value = 1050.0
    new_mcap = 5000000000.0
    
    new_divisor = calculate_adjusted_divisor(new_mcap, last_index_value)
    
    # Recalculate index with new divisor
    calculated_index = new_mcap / new_divisor
    
    assert round(calculated_index, 6) == round(last_index_value, 6)

def test_divisor_adjustment_zero_division():
    """Verify that the function handles zero index value gracefully."""
    assert calculate_adjusted_divisor(1000, 0) == 0

def test_divisor_scaling():
    """Test that doubling the market cap results in a doubled divisor for the same index value."""
    last_index = 1000.0
    mcap1 = 1000000.0
    mcap2 = 2000000.0
    
    div1 = calculate_adjusted_divisor(mcap1, last_index)
    div2 = calculate_adjusted_divisor(mcap2, last_index)
    
    assert div2 == 2 * div1


def test_calculate_daily_index_returns_when_no_trading_days(monkeypatch):
    monkeypatch.setattr(construction.pd, "read_sql", lambda *args, **kwargs: pd.DataFrame())

    writes = []

    def fake_to_sql(self, *args, **kwargs):
        writes.append(self.copy())

    monkeypatch.setattr(construction.pd.DataFrame, "to_sql", fake_to_sql, raising=False)

    construction.calculate_daily_index(dt.date(2024, 1, 1), dt.date(2024, 1, 2))

    assert writes == []


def test_calculate_daily_index_handles_rebalance_and_standard_day(monkeypatch):
    d1 = dt.date(2024, 1, 2)
    d2 = dt.date(2024, 1, 3)
    d3 = dt.date(2024, 1, 4)

    class DummyConn:
        def __init__(self):
            self.executed = []

        def execute(self, statement, params):
            self.executed.append((str(statement), params))

    class DummyBegin:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            return self._conn

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyEngine:
        def __init__(self, conn):
            self._conn = conn

        def begin(self):
            return DummyBegin(self._conn)

    conn = DummyConn()
    monkeypatch.setattr(construction, "engine", DummyEngine(conn))

    def fake_read_sql(query, _engine, params=None):
        sql = str(query)
        if "SELECT DISTINCT date FROM daily_prices" in sql:
            return pd.DataFrame({"date": [d1, d2, d3]})
        if "FROM index_constituents" in sql:
            t_date = params["t_date"]
            if t_date in (d1, d2):
                return pd.DataFrame(
                    {"ticker": ["AAA"], "shares_in_index": [10.0], "rebalance_date": [d1]}
                )
            return pd.DataFrame(
                {"ticker": ["AAA"], "shares_in_index": [20.0], "rebalance_date": [d3]}
            )
        if "FROM daily_prices" in sql and "ticker IN" in sql:
            t_date = params["t_date"]
            close_map = {d1: 100.0, d2: 110.0, d3: 110.0}
            return pd.DataFrame({"ticker": ["AAA"], "close_price": [close_map[t_date]]})
        raise AssertionError(f"Unexpected query: {sql}")

    written = {}

    def fake_to_sql(self, name, _engine, if_exists="fail", index=True):
        written["name"] = name
        written["if_exists"] = if_exists
        written["index"] = index
        written["df"] = self.copy()

    monkeypatch.setattr(construction.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(construction.pd.DataFrame, "to_sql", fake_to_sql, raising=False)

    construction.calculate_daily_index(d1, d3)

    assert conn.executed
    assert written["name"] == "index_values"
    assert written["if_exists"] == "append"
    result_df = written["df"]
    assert len(result_df) == 3
    assert result_df["value"].tolist() == [1000.0, 1100.0, 1100.0]
    assert result_df["divisor"].tolist() == [1.0, 1.0, 2.0]
