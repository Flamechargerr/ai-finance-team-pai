import datetime as dt
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import attribution, performance, visualizer
from index_math import rebalancing


def test_get_mock_free_float_is_stable_and_bounded():
    first = rebalancing.get_mock_free_float("AAPL")
    second = rebalancing.get_mock_free_float("AAPL")
    assert first == second
    assert 0.70 <= first <= 1.0


def test_compare_vs_benchmark_uses_synthetic_when_download_fails(monkeypatch):
    index_df = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)],
            "index_value": [1000.0, 1010.0, 1020.0],
        }
    )

    monkeypatch.setattr(performance.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(performance.pd, "read_sql", lambda *args, **kwargs: index_df.copy())

    result = performance.compare_vs_benchmark(dt.date(2024, 1, 2), dt.date(2024, 1, 4), "SPY")

    assert not result.empty
    assert "bench_rebased" in result.columns
    assert result["bench_rebased"].iloc[0] == 1000.0
    assert "index_return" in result.columns
    assert "bench_return" in result.columns


def test_compare_vs_benchmark_returns_empty_when_no_index_points(monkeypatch):
    monkeypatch.setattr(performance.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(performance.pd, "read_sql", lambda *args, **kwargs: pd.DataFrame())

    result = performance.compare_vs_benchmark(dt.date(2024, 1, 2), dt.date(2024, 1, 4), "SPY")
    assert result.empty


def test_calculate_turnover(monkeypatch):
    source_df = pd.DataFrame(
        {
            "rebalance_date": [dt.date(2024, 5, 31), dt.date(2024, 5, 31), dt.date(2024, 11, 30), dt.date(2024, 11, 30)],
            "ticker": ["AAA", "BBB", "AAA", "CCC"],
            "weight": [0.6, 0.4, 0.5, 0.5],
        }
    )
    monkeypatch.setattr(attribution.pd, "read_sql", lambda *args, **kwargs: source_df.copy())

    result = attribution.calculate_turnover()

    assert len(result) == 1
    assert result["rebalance_date"].iloc[0] == dt.date(2024, 11, 30)
    assert result["turnover_pct"].iloc[0] == 50.0


def test_calculate_sector_drift_pivots_to_percentages(monkeypatch):
    source_df = pd.DataFrame(
        {
            "rebalance_date": [dt.date(2024, 5, 31), dt.date(2024, 5, 31)],
            "sector": ["Tech", "Health"],
            "sector_weight": [0.7, 0.3],
        }
    )
    monkeypatch.setattr(attribution.pd, "read_sql", lambda *args, **kwargs: source_df.copy())

    result = attribution.calculate_sector_drift()

    assert result.loc[dt.date(2024, 5, 31), "Tech"] == 70.0
    assert result.loc[dt.date(2024, 5, 31), "Health"] == 30.0


def test_plot_performance_creates_image(tmp_path):
    output = tmp_path / "index_performance.png"
    perf_df = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "index_value": [1000.0, 1015.0],
            "bench_rebased": [1000.0, 1008.0],
        }
    )

    visualizer.plot_performance(perf_df, str(output))

    assert output.exists()


def test_calculate_rebalance_returns_empty_when_liquidity_filter_excludes_all(monkeypatch):
    universe_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "shares_outstanding": [100, 200],
            "close_price": [10.0, 20.0],
            "adtv_20d": [1000, 2000],
        }
    )

    def fake_read_sql(query, *_args, **_kwargs):
        sql = str(query)
        if "FROM equities e" in sql:
            return universe_df.copy()
        if "SELECT ticker FROM index_constituents" in sql:
            return pd.DataFrame({"ticker": []})
        raise AssertionError(f"Unexpected query: {sql}")

    monkeypatch.setattr(rebalancing.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(rebalancing, "get_mock_free_float", lambda _ticker: 1.0)

    result = rebalancing.calculate_rebalance(dt.date(2024, 5, 31))

    assert result.empty
    assert list(result.columns) == [
        "ticker",
        "rebalance_date",
        "weight",
        "shares_in_index",
        "free_float_factor",
    ]


def test_calculate_rebalance_retains_constituent_on_upper_buffer_boundary(monkeypatch):
    universe_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "shares_outstanding": [100, 100, 100, 100, 100],
            "close_price": [100.0, 90.0, 80.0, 70.0, 60.0],
            "adtv_20d": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000],
        }
    )

    def fake_read_sql(query, *_args, **_kwargs):
        sql = str(query)
        if "FROM equities e" in sql:
            return universe_df.copy()
        if "SELECT ticker FROM index_constituents" in sql:
            return pd.DataFrame({"ticker": ["DDD"]})
        raise AssertionError(f"Unexpected query: {sql}")

    monkeypatch.setattr(rebalancing.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(rebalancing, "get_mock_free_float", lambda _ticker: 1.0)
    monkeypatch.setattr(rebalancing, "TARGET_CONSTITUENTS", 3)
    monkeypatch.setattr(rebalancing, "BUFFER_ZONE", 1)

    result = rebalancing.calculate_rebalance(dt.date(2024, 5, 31))
    selected = set(result["ticker"].tolist())

    assert selected == {"AAA", "BBB", "DDD"}
    assert "CCC" not in selected
    assert round(result["weight"].sum(), 10) == 1.0
