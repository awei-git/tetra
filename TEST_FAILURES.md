# Test Failures Documentation

This document records known test failures, their root causes, and any relevant context.

---

## 1. `tests/test_database.py`

**Status:** Failing

**Description:** Tests for the database schema and TimescaleDB integration.

**Failures:**
*   `test_ohlcv_is_hypertable`: Fails because `market_data.ohlcv` is not recognized as a TimescaleDB hypertable.
    *   **Error:** `AssertionError: market_data.ohlcv is not a hypertable.`
    *   **Root Cause:** The `create_hypertable` command might not have been executed or applied correctly during database setup, or the TimescaleDB extension is not properly enabled/recognized.
    *   **Test Purpose/Intent:** Verify that the `ohlcv` table is correctly configured as a TimescaleDB hypertable, which is crucial for time-series data performance.
    *   **Test Setup:** Connects to the real PostgreSQL database using `sqlalchemy` and queries `timescaledb_information.hypertables`.
    *   **Expected Outcome:** The query should return a row indicating `market_data.ohlcv` is a hypertable.
    *   **Relevant Code Snippet:**
        ```python
        def test_ohlcv_is_hypertable(engine):
            with engine.connect() as connection:
                query = text("""
                    SELECT 1
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'market_data'
                    AND hypertable_name = 'ohlcv';
                """)
                result = connection.execute(query).scalar_one_or_none()
                assert result == 1, "market_data.ohlcv is not a hypertable."
        ```
*   `test_economic_data_table_columns`: Fails due to column discrepancies in `economic_data.economic_data`.
    *   **Error:** `AssertionError: assert ['id', 'symbo...iminary', ...] == ['id', 'symbo...on_date', ...]`
    *   **Root Cause:** The actual table schema does not match the expected schema defined in the test and `docs/DATABASE.md`. Specifically, `revision_date` and `is_preliminary` columns are swapped or mismatched.
    *   **Test Purpose/Intent:** Verify that the `economic_data.economic_data` table has the correct column names and order as per the schema definition.
    *   **Test Setup:** Connects to the real PostgreSQL database and uses `inspector.get_columns` to retrieve column information.
    *   **Expected Outcome:** The list of column names should exactly match the `expected_columns` list.
    *   **Relevant Code Snippet:**
        ```python
        def test_economic_data_table_columns(engine):
            inspector = inspect(engine)
            columns = inspector.get_columns("economic_data", schema="economic_data")
            column_names = [col["name"] for col in columns]
            expected_columns = [
                "id", "symbol", "date", "value", "is_preliminary", "revision_date",
                "source", "created_at", "updated_at"
            ]
            assert column_names == expected_columns
        ```
*   `test_news_articles_table_columns`: Fails due to column discrepancies in `news.news_articles`.
    *   **Error:** `AssertionError: assert ['id', 'sourc... 'title', ...] == ['id', 'title...'source', ...]`
    *   **Root Cause:** The actual table schema does not match the expected schema. Columns like `source_id` and `image_url` might be present in the database but not expected by the test, or vice-versa.
    *   **Test Purpose/Intent:** Verify that the `news.news_articles` table has the correct column names and order.
    *   **Test Setup:** Connects to the real PostgreSQL database and uses `inspector.get_columns`.
    *   **Expected Outcome:** The list of column names should exactly match the `expected_columns` list.
    *   **Relevant Code Snippet:**
        ```python
        def test_news_articles_table_columns(engine):
            inspector = inspect(engine)
            columns = inspector.get_columns("news_articles", schema="news")
            column_names = [col["name"] for col in columns]
            expected_columns = [
                "id", "title", "description", "content", "url", "source", "author",
                "published_at", "created_at"
            ]
            assert column_names == expected_columns
        ```
*   `test_event_data_table_columns`: Fails due to column discrepancies in `events.event_data`.
    *   **Error:** `AssertionError: assert ['id', 'event...'impact', ...] == ['id', 'event...t_level', ...]`
    *   **Root Cause:** The actual table schema does not match the expected schema. Specifically, `impact` and `impact_level` are mismatched, and `updated_at` might be an unexpected column.
    *   **Test Purpose/Intent:** Verify that the `events.event_data` table has the correct column names and order.
    *   **Test Setup:** Connects to the real PostgreSQL database and uses `inspector.get_columns`.
    *   **Expected Outcome:** The list of column names should exactly match the `expected_columns` list.
    *   **Relevant Code Snippet:**
        ```python
        def test_event_data_table_columns(engine):
            inspector = inspect(engine)
            columns = inspector.get_columns("event_data", schema="events")
            column_names = [col["name"] for col in columns]
            expected_columns = [
                "id", "event_type", "event_datetime", "event_name", "description",
                "impact_level", "symbol", "currency", "country", "actual",
                "forecast", "previous", "source", "created_at"
            ]
            assert column_names == expected_columns
        ```

**Overall Root Cause:** Discrepancies between the expected database schema (as per `docs/DATABASE.md` and the tests) and the actual deployed database schema. This suggests issues with database migration scripts or the initial `init.sql` setup.

---

## 2. `tests/test_data_pipeline_final.py` (DataQualityCheckStep)

**Status:** Failing (Mocking Challenges)

**Description:** Unit tests for the `DataQualityCheckStep` using a mocked database.

**Test Purpose/Intent:** To verify that the `DataQualityCheckStep` correctly identifies and reports various data quality issues (stale data, missing symbols, data gaps) based on mocked database query results.

**Test Setup:**
*   Uses `pytest.mark.parametrize` to run a single test function (`test_data_quality_check_step`) with different scenarios (`good_data`, `stale_data`, `missing_symbols`, `data_gaps`).
*   Patches `src.pipelines.data_pipeline.steps.data_quality.get_session` to return a custom `mock_db_session`.
*   The `mock_db_session.execute` method is mocked with an `async def mock_execute` function. This function inspects the `query.text` to determine which predefined `mock_data` (e.g., `symbol_stats`, `gap_query`) to return for `fetchall` or `fetchone` calls.

**Expected Outcome:** For each scenario, the test expects specific counts for `stale_data`, `missing_symbols`, and `market_data_gaps` in the `result` dictionary returned by `DataQualityCheckStep.execute`.

**Relevant Code Snippet (from `tests/test_data_pipeline_final.py`):**
```python
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario, symbols, mock_data, expected_stale, expected_missing, expected_gaps", [
    (
        "good_data",
        ["AAPL"],
        {
            "symbol_stats": [MagicMock(symbol='AAPL', record_count=100, first_date=date(2023, 1, 1), last_date=date.today(), days_with_data=200)],
            "gap_query": [],
            "coverage_stats": MagicMock(unique_symbols=1, total_records=100, earliest_date=date(2023,1,1), latest_date=date.today(), unique_days=200, recent_records_7d=5),
            "econ_coverage": MagicMock(indicators=10, total_records=1000, earliest_date=date(2023,1,1), latest_date=date.today()),
            "news_coverage": MagicMock(total_articles=1000, unique_sources=10, earliest_date=date(2023,1,1), latest_date=date.today(), recent_articles_7d=50),
            "event_coverage": [MagicMock(event_type='earnings', count=100, earliest_date=date(2023,1,1), latest_date=date.today())],
            "summary": MagicMock(unique_symbols=1, total_records=100, earliest_data=date(2023,1,1), latest_data=date.today())
        },
        0, 0, 0
    ),
    # ... other scenarios ...
])
@patch('src.pipelines.data_pipeline.steps.data_quality.get_session')
async def test_data_quality_check_step(mock_get_session, data_quality_check_step, scenario, symbols, mock_data, expected_stale, expected_missing, expected_gaps):
    async def mock_execute(query, params=None):
        result = AsyncMock()
        query_text = query.text
        # ... logic to return mock_data based on query_text ...
        return result

    mock_db_session = AsyncMock()
    mock_db_session.execute = mock_execute
    mock_get_session.return_value.__aenter__.return_value = mock_db_session

    context = PipelineContext()
    context.data = {
        "symbols": symbols,
        "mode": "daily"
    }

    result = await data_quality_check_step.execute(context)

    assert len(result["stale_data"]) == expected_stale
    assert len(result["missing_symbols"]) == expected_missing
    assert len(result["market_data_gaps"]) == expected_gaps
```

**Failures:**
*   `test_data_quality_check_step[stale_data-...]`
*   `test_data_quality_check_step[missing_symbols-...]`
*   `test_data_quality_check_step[data_gaps-...]`

**Root Cause:** The primary challenge lies in effectively mocking the complex and conditional database interactions within the `DataQualityCheckStep`. The step performs multiple SQL queries, and the results of these queries influence subsequent logic. Despite several attempts to refine the mocking strategy (using `side_effect` with `AsyncMock` and custom `mock_execute` functions), the mock database has not consistently replicated the behavior required to trigger the expected data quality issues (stale data, missing symbols, gaps) within the test environment. The tests assert that `len(result[...]) == expected_count`, but the actual count remains 0, indicating the internal logic for identifying these issues is not being reached or correctly evaluated by the mock.

---

## 3. `tests/simulators/test_historical_simulator_extended.py`

**Status:** Failing (Missing Module Dependency)

**Description:** Integration test for the `HistoricalSimulator` with a sample strategy.

**Test Purpose/Intent:** To verify that the `HistoricalSimulator` can successfully run a simulation with a given strategy, process market data, and produce a simulation result. It specifically aims to test the integration with a strategy that relies on technical indicators.

**Test Setup:**
*   Loads an RSI Mean Reversion strategy from a YAML file.
*   Patches `src.simulators.historical.simulator.MarketReplay` to mock market data loading and retrieval.
*   Patches `src.simulators.historical.simulator.TechnicalIndicators` to prevent `ModuleNotFoundError` (as `src/ml` does not exist) and to control the output of indicator calculations.
*   Provides a mock `pandas.DataFrame` as OHLCV data for the `MarketReplay` mock.

**Expected Outcome:** The simulation should run without errors, and the `SimulationResult` object should be returned with a positive `final_value`.

**Relevant Code Snippet (from `tests/simulators/test_historical_simulator_extended.py`):**
```python
@pytest.fixture
def rsi_strategy():
    return StrategyConfigLoader.load_from_file("/Users/angwei/Repos/tetra/src/strats/examples/signal_based.yaml")

@pytest.mark.asyncio
@patch('src.simulators.historical.simulator.MarketReplay')
@patch('src.simulators.historical.simulator.TechnicalIndicators')
async def test_historical_simulator_with_rsi_strategy(MockTechnicalIndicators, MockMarketReplay, rsi_strategy):
    mock_market_replay = MockMarketReplay.return_value
    mock_market_replay.load_data = AsyncMock()
    mock_data = pd.DataFrame({ # ... OHLCV data ... })
    mock_market_replay.get_market_data = AsyncMock(side_effect=lambda symbols, trading_day: {symbol: mock_data.loc[pd.to_datetime(trading_day)].to_dict() for symbol in symbols})
    mock_market_replay._load_symbol_data = AsyncMock(return_value=mock_data)

    MockTechnicalIndicators.calculate_all_indicators.return_value = mock_data

    simulator = HistoricalSimulator()
    portfolio = Portfolio(initial_cash=100000)

    result = await simulator.run_simulation(
        portfolio=portfolio,
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 6),
        strategy=rsi_strategy
    )

    assert result is not None
    assert result.final_value > 0
```

**Failures:**
*   `test_historical_simulator_with_rsi_strategy`

**Error:** `AttributeError: <module 'src.simulators.historical.simulator' from '/Users/angwei/Repos/tetra/src/simulators/historical/simulator.py'> does not have the attribute 'TechnicalIndicators'`

**Root Cause:** The `HistoricalSimulator` (specifically its `_execute_strategy` method) attempts to import `TechnicalIndicators` from `src.ml.technical_indicators`. However, the `src/ml` directory and its contents (including `technical_indicators.py`) do not exist in the current project structure. This is a hard dependency on a non-existent module. Attempts to patch this import within the test environment have been unsuccessful because the import occurs locally within a method, making it difficult to mock without modifying the `HistoricalSimulator`'s source code.

---