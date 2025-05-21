import pytest
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
from datetime import datetime, date, timedelta

from src.tools import api as financial_api
from src.data.models import Price, FinancialMetrics # Added FinancialMetrics
# Assuming Cache is imported from src.data.cache as used in api.py
# from src.data.cache import Cache
# However, api.py uses `_cache = get_cache()`, so direct Cache interaction might not be needed in tests
# if we mock `get_cache` or the cache instance itself.

# Remove placeholder if it's the only test, or keep if adding more unrelated tests later
# def test_example_placeholder():
#     assert True

# Test Data Fixture for yfinance history
@pytest.fixture
def sample_history_df():
    return pd.DataFrame({
        'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
        'High': [152.5, 153.5, 154.5, 155.5, 156.5],
        'Low': [149.5, 150.5, 151.5, 152.5, 153.5],
        'Close': [151.5, 152.5, 153.5, 154.5, 155.5],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_successful_fetch_and_parse(mock_yf_ticker, mock_cache, sample_history_df):
    # Arrange
    mock_cache.get_prices.return_value = None # Simulate cache miss

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_history_df
    mock_yf_ticker.return_value = mock_ticker_instance

    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-03" # Requesting 3 days from sample_history_df

    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    assert len(result) == 3
    assert all(isinstance(p, Price) for p in result)

    assert result[0].time == "2023-01-01"
    assert result[0].open == 150.0
    assert result[0].high == 152.5
    assert result[0].low == 149.5
    assert result[0].close == 151.5
    assert result[0].volume == 1000000
    assert isinstance(result[0].volume, int)
    assert isinstance(result[0].open, float)

    assert result[2].time == "2023-01-03"
    assert result[2].open == 152.0
    assert result[2].volume == 1200000

    # Check yfinance call arguments (end_date is adjusted by +1 day)
    expected_yf_end_date = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    mock_ticker_instance.history.assert_called_once_with(start=start_date, end=expected_yf_end_date, interval="1d")
    
    # Check cache set call
    expected_cache_key = f"prices_{ticker}_{start_date}_{end_date}"
    mock_cache.set_prices.assert_called_once()
    args, _ = mock_cache.set_prices.call_args
    assert args[0] == expected_cache_key
    assert len(args[1]) == 3 # 3 items should be cached
    assert args[1][0]['time'] == "2023-01-01"


@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_date_range_filtering(mock_yf_ticker, mock_cache, sample_history_df):
    # Arrange
    mock_cache.get_prices.return_value = None # Cache miss
    
    mock_ticker_instance = MagicMock()
    # yf.history might return more data than requested if start/end are broad,
    # but our function also filters strictly.
    # Here, we simulate yf.history returning the full sample_history_df for the adjusted dates.
    mock_ticker_instance.history.return_value = sample_history_df
    mock_yf_ticker.return_value = mock_ticker_instance

    ticker = "AAPL"
    start_date = "2023-01-02"
    end_date = "2023-01-04" # Requesting 3 specific days

    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    assert len(result) == 3
    assert result[0].time == "2023-01-02"
    assert result[1].time == "2023-01-03"
    assert result[2].time == "2023-01-04"

    # Ensure yf.Ticker().history was called (once)
    mock_ticker_instance.history.assert_called_once()
    # Cache should be set with the filtered results
    expected_cache_key = f"prices_{ticker}_{start_date}_{end_date}"
    mock_cache.set_prices.assert_called_once()
    args, _ = mock_cache.set_prices.call_args
    assert args[0] == expected_cache_key
    assert len(args[1]) == 3

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_from_cache(mock_yf_ticker, mock_cache, sample_history_df):
    # Arrange
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-02"
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"

    # Prepare cached data (as list of dicts, which is how it's stored)
    # These are already filtered for the specific date range by the key.
    cached_data_dicts = [
        Price(time="2023-01-01", open=150.0, high=152.5, low=149.5, close=151.5, volume=1000000).model_dump(),
        Price(time="2023-01-02", open=151.0, high=153.5, low=150.5, close=152.5, volume=1100000).model_dump(),
    ]
    mock_cache.get_prices.return_value = cached_data_dicts
    
    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    mock_cache.get_prices.assert_called_once_with(cache_key)
    mock_yf_ticker.assert_not_called() # yfinance should not be called
    
    assert len(result) == 2
    assert result[0].time == "2023-01-01"
    assert result[0].open == 150.0
    assert isinstance(result[0].volume, int) # Check type consistency from cache
    assert result[1].time == "2023-01-02"
    assert result[1].volume == 1100000

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_data_not_in_cache_then_fetched_and_cached(mock_yf_ticker, mock_cache, sample_history_df):
    # Arrange
    ticker = "MSFT"
    start_date = "2023-01-03"
    end_date = "2023-01-03"
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"

    mock_cache.get_prices.return_value = None # Simulate cache miss

    # yf.history will return the full sample_history_df, function will filter
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_history_df 
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    mock_cache.get_prices.assert_called_once_with(cache_key)
    mock_yf_ticker.assert_called_once_with(ticker)
    mock_ticker_instance.history.assert_called_once()

    assert len(result) == 1
    assert result[0].time == "2023-01-03"
    assert result[0].open == 152.0
    assert result[0].volume == 1200000

    # Verify data was set to cache
    expected_cached_data = [result[0].model_dump()] # Data for 2023-01-03
    mock_cache.set_prices.assert_called_once_with(cache_key, expected_cached_data)


@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_invalid_ticker_or_empty_history(mock_yf_ticker, mock_cache):
    # Arrange
    ticker = "INVALIDTICKER"
    start_date = "2023-01-01"
    end_date = "2023-01-01"
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"

    mock_cache.get_prices.return_value = None # Cache miss

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = pd.DataFrame() # Empty DataFrame
    mock_yf_ticker.return_value = mock_ticker_instance

    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    assert result == []
    mock_cache.get_prices.assert_called_once_with(cache_key)
    mock_yf_ticker.assert_called_once_with(ticker)
    mock_ticker_instance.history.assert_called_once()
    # Ensure it caches the empty result for this specific key
    mock_cache.set_prices.assert_called_once_with(cache_key, [])

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_prices_yfinance_api_error(mock_yf_ticker, mock_cache):
    # Arrange
    ticker = "ERRORTICKER"
    start_date = "2023-01-01"
    end_date = "2023-01-01"
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"

    mock_cache.get_prices.return_value = None # Cache miss

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = Exception("Simulated yfinance API error")
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Act
    result = financial_api.get_prices(ticker, start_date, end_date)

    # Assert
    assert result == []
    mock_cache.get_prices.assert_called_once_with(cache_key)
    mock_yf_ticker.assert_called_once_with(ticker)
    mock_ticker_instance.history.assert_called_once()
    mock_cache.set_prices.assert_not_called() # Should not cache if error occurs during fetch


# --- Tests for get_market_cap ---

@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_successful_fetch(mock_yf_ticker):
    # Arrange
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'marketCap': 2_000_000_000_000.0}
    mock_yf_ticker.return_value = mock_ticker_instance
    
    ticker = "AAPL"
    # For this test, let end_date be today to avoid the warning, simplifying the assertion
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = financial_api.get_market_cap(ticker, end_date)

    # Assert
    assert result == 2_000_000_000_000.0
    assert isinstance(result, float)
    mock_yf_ticker.assert_called_once_with(ticker)

@patch('src.tools.api.datetime') # Mocking datetime within the api.py module
@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_historical_end_date_prints_warning(mock_yf_ticker, mock_api_datetime, capsys):
    # Arrange
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'marketCap': 2.5e12} # 2.5 Trillion
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Mock datetime.datetime.now() within the context of src.tools.api
    mock_api_datetime.datetime.now.return_value = datetime(2023, 1, 10) # "Today" is Jan 10, 2023

    ticker = "MSFT"
    historical_end_date = "2023-01-01" # A date before "today"

    # Act
    result = financial_api.get_market_cap(ticker, historical_end_date)
    captured_stdout = capsys.readouterr().out

    # Assert
    assert result == 2.5e12
    expected_warning = (f"Warning: get_market_cap for ticker {ticker} called with end_date {historical_end_date}. "
                        f"This function returns the CURRENT market cap from yfinance, not historical.")
    assert expected_warning in captured_stdout

@patch('src.tools.api.datetime') # Mocking datetime within the api.py module
@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_current_end_date_no_warning(mock_yf_ticker, mock_api_datetime, capsys):
    # Arrange
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'marketCap': 1.5e12}
    mock_yf_ticker.return_value = mock_ticker_instance

    current_date_str = "2023-01-10"
    mock_api_datetime.datetime.now.return_value = datetime.strptime(current_date_str, "%Y-%m-%d")

    ticker = "GOOG"
    end_date = current_date_str # end_date is "today"

    # Act
    result = financial_api.get_market_cap(ticker, end_date)
    captured_stdout = capsys.readouterr().out
    
    # Assert
    assert result == 1.5e12
    assert "Warning:" not in captured_stdout

@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_missing_in_info(mock_yf_ticker, capsys):
    # Arrange
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'someOtherData': 123} # marketCap is missing
    mock_yf_ticker.return_value = mock_ticker_instance
    
    ticker = "XYZ"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = financial_api.get_market_cap(ticker, end_date)
    captured_stdout = capsys.readouterr().out

    # Assert
    assert result is None
    assert f"Market cap not found for ticker {ticker} in yfinance info." in captured_stdout

@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_info_is_none(mock_yf_ticker, capsys):
    # Arrange
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = None # info itself is None
    mock_yf_ticker.return_value = mock_ticker_instance
    
    ticker = "NULLINFO"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = financial_api.get_market_cap(ticker, end_date)
    captured_stdout = capsys.readouterr().out
    
    # Assert
    # The .get("marketCap") on a None info dict will raise an AttributeError, caught by the general Exception
    assert result is None
    assert f"Error fetching market cap for {ticker} using yfinance:" in captured_stdout


# --- Helper data for get_financial_metrics tests ---

@pytest.fixture
def sample_yf_info_ttm():
    return {
        'currency': 'USD',
        'marketCap': 2.0e12,
        'enterpriseValue': 2.1e12,
        'trailingPE': 25.0,
        'priceToBook': 5.0,
        'priceToSalesTrailing12Months': 4.0,
        'enterpriseToEbitda': 20.0,
        'enterpriseToRevenue': 4.5,
        'pegRatio': 1.8,
        'grossMargins': 0.6,
        'operatingMargins': 0.3,
        'profitMargins': 0.25,
        'returnOnEquity': 0.22,
        'returnOnAssets': 0.10,
        'payoutRatio': 0.3,
        'trailingEps': 6.0,
        'revenueGrowth': 0.15, # quarter over quarter if from info
        'earningsQuarterlyGrowth': 0.18, # from info
        'currentRatio': 2.0,
        'quickRatio': 1.5,
        'cashRatio': 1.0,
        'debtToEquity': 50.0, # yfinance often gives this as a percentage
        'bookValue': 30.0, # per share
        'sharesOutstanding': 16e9
    }

@pytest.fixture
def sample_quarterly_cashflow_df():
    # For TTM FCF calculation (OpCash - CapEx)
    # Assuming CapEx is reported as negative by yfinance if it's an outflow, or positive if it needs subtraction
    return pd.DataFrame({
        pd.to_datetime('2023-03-31'): {'Total Cash From Operating Activities': 20e9, 'Capital Expenditures': -5e9, 'Free Cash Flow': 15e9},
        pd.to_datetime('2022-12-31'): {'Total Cash From Operating Activities': 22e9, 'Capital Expenditures': -6e9, 'Free Cash Flow': 16e9},
        pd.to_datetime('2022-09-30'): {'Total Cash From Operating Activities': 18e9, 'Capital Expenditures': -4e9, 'Free Cash Flow': 14e9},
        pd.to_datetime('2022-06-30'): {'Total Cash From Operating Activities': 19e9, 'Capital Expenditures': -5.5e9, 'Free Cash Flow': 13.5e9},
        pd.to_datetime('2022-03-31'): {'Total Cash From Operating Activities': 17e9, 'Capital Expenditures': -3.5e9, 'Free Cash Flow': 13.5e9},
    }).T # Transposed to match yfinance structure (dates as columns)

@pytest.fixture
def sample_annual_financials_df():
    return pd.DataFrame({
        pd.to_datetime('2022-12-31'): {'Total Revenue': 100e9, 'Gross Profit': 60e9, 'Operating Income': 30e9, 'Net Income': 20e9, 'Basic EPS': 5.0, 'EBITDA': 35e9},
        pd.to_datetime('2021-12-31'): {'Total Revenue': 90e9, 'Gross Profit': 50e9, 'Operating Income': 25e9, 'Net Income': 15e9, 'Basic EPS': 4.0, 'EBITDA': 30e9},
    }).T

@pytest.fixture
def sample_annual_balance_sheet_df():
    return pd.DataFrame({
        pd.to_datetime('2022-12-31'): {'Total Assets': 200e9, 'Total Liab': 100e9, 'Total Stockholder Equity': 100e9, 'Total Current Assets': 80e9, 'Total Current Liabilities': 40e9, 'Cash': 10e9, 'Inventory': 5e9},
        pd.to_datetime('2021-12-31'): {'Total Assets': 180e9, 'Total Liab': 90e9, 'Total Stockholder Equity': 90e9, 'Total Current Assets': 70e9, 'Total Current Liabilities': 35e9, 'Cash': 8e9, 'Inventory': 4e9},
    }).T

@pytest.fixture
def sample_annual_cashflow_df():
    return pd.DataFrame({
        pd.to_datetime('2022-12-31'): {'Total Cash From Operating Activities': 25e9, 'Capital Expenditures': -10e9, 'Free Cash Flow': 15e9},
        pd.to_datetime('2021-12-31'): {'Total Cash From Operating Activities': 22e9, 'Capital Expenditures': -8e9, 'Free Cash Flow': 14e9},
    }).T


# --- Tests for get_financial_metrics ---

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_ttm_successful_fetch(mock_yf_ticker, mock_cache, sample_yf_info_ttm, sample_quarterly_cashflow_df):
    # Arrange
    ticker = "TTMSTOCK"
    end_date = "2023-03-31" # Should match latest quarter in sample_quarterly_cashflow_df for TTM FCF
    period = "ttm"
    limit = 1 # TTM usually returns 1 result

    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    mock_cache.get_financial_metrics.return_value = None # Cache miss

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = sample_yf_info_ttm
    mock_ticker_instance.quarterly_cashflow = sample_quarterly_cashflow_df.T # .T because yf returns dates as columns
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert len(results) == 1
    metric = results[0]
    assert isinstance(metric, FinancialMetrics)
    assert metric.ticker == ticker
    assert metric.period == "ttm"
    assert metric.report_period == end_date # TTM report_period is the end_date specified
    assert metric.currency == "USD"
    assert metric.market_cap == 2.0e12
    assert metric.trailingPE == 25.0
    assert metric.profitMargins == 0.25 # check one alias mapping from yf
    
    # Check TTM FCF related calculation (15+16+14+13.5)e9 = 58.5e9
    # FCF Yield = 58.5e9 / 2.0e12 = 0.02925
    # FCF Per Share = 58.5e9 / 16e9 = 3.65625
    expected_fcf_ttm = 58.5e9 
    assert metric.free_cash_flow_yield == pytest.approx(expected_fcf_ttm / 2.0e12)
    assert metric.free_cash_flow_per_share == pytest.approx(expected_fcf_ttm / 16e9)

    mock_cache.set_financial_metrics.assert_called_once_with(cache_key, [metric.model_dump()])

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_annual_successful_fetch(
    mock_yf_ticker, mock_cache, 
    sample_annual_financials_df, sample_annual_balance_sheet_df, sample_annual_cashflow_df
):
    # Arrange
    ticker = "ANNUALSTOCK"
    end_date = "2022-12-31"
    period = "annual"
    limit = 2

    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    mock_cache.get_financial_metrics.return_value = None

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'currency': 'CAD'} # Test different currency
    mock_ticker_instance.financials = sample_annual_financials_df.T
    mock_ticker_instance.balance_sheet = sample_annual_balance_sheet_df.T
    mock_ticker_instance.cashflow = sample_annual_cashflow_df.T
    mock_yf_ticker.return_value = mock_ticker_instance

    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert len(results) == 2 # Based on limit and available data
    metric_2022 = results[0]
    metric_2021 = results[1]

    assert metric_2022.report_period == "2022-12-31"
    assert metric_2022.period == "annual"
    assert metric_2022.currency == "CAD"
    assert metric_2022.total_revenue == 100e9
    assert metric_2022.net_income == 20e9
    assert metric_2022.gross_margin == pytest.approx(60e9 / 100e9)
    assert metric_2022.current_ratio == pytest.approx(80e9 / 40e9)
    # Test a growth rate (2022 vs 2021)
    # Revenue growth: (100-90)/90 = 0.111...
    assert metric_2022.revenue_growth == pytest.approx((100e9 - 90e9) / 90e9)
    assert metric_2022.earnings_growth == pytest.approx((20e9 - 15e9) / 15e9)


    assert metric_2021.report_period == "2021-12-31"
    assert metric_2021.total_revenue == 90e9
    assert metric_2021.revenue_growth is None # No prior data for 2020 to calculate growth

    mock_cache.set_financial_metrics.assert_called_once()
    args, _ = mock_cache.set_financial_metrics.call_args
    assert args[0] == cache_key
    assert len(args[1]) == 2


@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_quarterly_end_date_and_limit(mock_yf_ticker, mock_cache):
    # Arrange
    ticker = "QTRSTOCK"
    end_date = "2023-06-30"
    period = "quarterly"
    limit = 2 # Expecting 2 results: 2023-06-30 and 2023-03-31

    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    mock_cache.get_financial_metrics.return_value = None

    q_fin_data = {
        pd.to_datetime('2023-06-30'): {'Total Revenue': 30e9},
        pd.to_datetime('2023-03-31'): {'Total Revenue': 28e9},
        pd.to_datetime('2022-12-31'): {'Total Revenue': 25e9}, # Should be filtered out by end_date or limit
    }
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'currency': 'USD'}
    # Create empty DFs for balance sheet and cashflow as they are expected by the function
    mock_ticker_instance.quarterly_financials = pd.DataFrame(q_fin_data).T # Transpose to match yf structure
    mock_ticker_instance.quarterly_balance_sheet = pd.DataFrame(index=q_fin_data.keys()) # Empty but with correct index
    mock_ticker_instance.quarterly_cashflow = pd.DataFrame(index=q_fin_data.keys())    # Empty but with correct index
    mock_yf_ticker.return_value = mock_ticker_instance
    
    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert len(results) == 2
    assert results[0].report_period == "2023-06-30"
    assert results[0].total_revenue == 30e9
    assert results[1].report_period == "2023-03-31"
    assert results[1].total_revenue == 28e9
    
    mock_cache.set_financial_metrics.assert_called_once_with(cache_key, [r.model_dump() for r in results])

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_from_cache(mock_yf_ticker, mock_cache):
    # Arrange
    ticker, period, end_date, limit = "CACHEHIT", "annual", "2022-12-31", 1
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    
    cached_metric_dict = FinancialMetrics(
        ticker=ticker, report_period="2022-12-31", period="annual", currency="USD", total_revenue=100e9
    ).model_dump()
    mock_cache.get_financial_metrics.return_value = [cached_metric_dict]

    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    mock_cache.get_financial_metrics.assert_called_once_with(cache_key)
    mock_yf_ticker.assert_not_called() # yfinance should not be called
    assert len(results) == 1
    assert results[0].total_revenue == 100e9

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_cache_deletion_on_error(mock_yf_ticker, mock_cache):
    # Arrange
    ticker, period, end_date, limit = "CACHEERROR", "annual", "2022-12-31", 1
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    
    mock_cache.get_financial_metrics.return_value = None # Cache miss
    mock_yf_ticker.side_effect = Exception("Simulated yfinance API error")

    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert results == []
    mock_cache.get_financial_metrics.assert_called_once_with(cache_key)
    mock_cache.set_financial_metrics.assert_not_called()
    mock_cache.delete_financial_metrics.assert_called_once_with(cache_key) # Cache should be deleted

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_empty_data_from_yfinance(mock_yf_ticker, mock_cache):
    # Arrange
    ticker, period, end_date, limit = "EMPTYDATA", "annual", "2022-12-31", 1
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    mock_cache.get_financial_metrics.return_value = None

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'currency': 'USD'}
    mock_ticker_instance.financials = pd.DataFrame() # Empty financials
    mock_ticker_instance.balance_sheet = pd.DataFrame()
    mock_ticker_instance.cashflow = pd.DataFrame()
    mock_yf_ticker.return_value = mock_ticker_instance

    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert results == []
    # Cache should still be set, but with an empty list, to avoid repeated calls for known empty data
    mock_cache.set_financial_metrics.assert_called_once_with(cache_key, [])

@patch('src.tools.api._cache')
@patch('src.tools.api.yf.Ticker')
def test_get_financial_metrics_insufficient_data_for_ttm_fcf(mock_yf_ticker, mock_cache, sample_yf_info_ttm):
    # Arrange
    ticker, period, end_date, limit = "PARTIALTTM", "ttm", "2023-03-31", 1
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}"
    mock_cache.get_financial_metrics.return_value = None

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = sample_yf_info_ttm
    # Only 2 quarters of cashflow data, not enough for TTM FCF sum
    partial_q_cashflow = pd.DataFrame({
        pd.to_datetime('2023-03-31'): {'Total Cash From Operating Activities': 20e9, 'Capital Expenditures': -5e9},
        pd.to_datetime('2022-12-31'): {'Total Cash From Operating Activities': 22e9, 'Capital Expenditures': -6e9},
    }).T
    mock_ticker_instance.quarterly_cashflow = partial_q_cashflow
    mock_yf_ticker.return_value = mock_ticker_instance

    # Act
    results = financial_api.get_financial_metrics(ticker, end_date, period, limit)

    # Assert
    assert len(results) == 1
    metric = results[0]
    assert metric.free_cash_flow_yield is None # FCF calculation should fail gracefully
    assert metric.free_cash_flow_per_share is None
    assert metric.market_cap == sample_yf_info_ttm['marketCap'] # Other fields from .info should be there
    mock_cache.set_financial_metrics.assert_called_once()

@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_yfinance_api_error_on_ticker(mock_yf_ticker, capsys):
    # Arrange
    mock_yf_ticker.side_effect = Exception("Simulated yfinance API error on Ticker instantiation")
    
    ticker = "ERRTICKER"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = financial_api.get_market_cap(ticker, end_date)
    captured_stdout = capsys.readouterr().out

    # Assert
    assert result is None
    assert f"Error fetching market cap for {ticker} using yfinance:" in captured_stdout

@patch('src.tools.api.yf.Ticker')
def test_get_market_cap_yfinance_api_error_on_info_access(mock_yf_ticker, capsys):
    # Arrange
    mock_ticker_instance = MagicMock()
    # Configure the .info attribute to raise an exception when accessed
    type(mock_ticker_instance).info = mock.PropertyMock(side_effect=Exception("Simulated error accessing .info"))
    mock_yf_ticker.return_value = mock_ticker_instance
    
    ticker = "INFOERR"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Act
    result = financial_api.get_market_cap(ticker, end_date)
    captured_stdout = capsys.readouterr().out

    # Assert
    assert result is None
    assert f"Error fetching market cap for {ticker} using yfinance:" in captured_stdout
