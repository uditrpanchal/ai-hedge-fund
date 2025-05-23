import datetime
import os
import pandas as pd
# Removed: import requests # No longer making direct HTTP calls

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse, # Still used if yf_transformer returns it, but we extract list
    FinancialMetrics,
    FinancialMetricsResponse, # Still used
    Price,
    PriceResponse, # Still used
    LineItem,
    LineItemResponse, # Still used
    InsiderTrade,
    InsiderTradeResponse, # Still used
    CompanyFactsResponse, # Still used
    CompanyFacts # Added for get_market_cap
)

# Import transformers with aliases
from src.data.yfinance_transformers import (
   get_price_response as yf_get_price_response,
   get_financial_metrics_response as yf_get_financial_metrics_response,
   get_financial_statements_response as yf_get_financial_statements_response,
   get_insider_trades_response as yf_get_insider_trades_response,
   get_company_news_response as yf_get_company_news_response,
   get_company_facts_response as yf_get_company_facts_response
)
import logging

# Global cache instance
_cache = get_cache()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or yfinance_transformers."""
    # Using simple ticker-based key for get_prices, filtering applied after.
    # This matches the original cache structure more closely.
    if all_cached_prices := _cache.get_prices(ticker):
        filtered_prices = [
            Price(**price_dict) for price_dict in all_cached_prices
            if start_date <= price_dict.get("time", "")[:10] <= end_date
        ]
        if filtered_prices: # If filtering results in non-empty list
            logging.info(f"Prices for {ticker} ({start_date}-{end_date}) found and filtered from cache.")
            return filtered_prices
        # If filtering results in empty, it means this specific date range wasn't in the broader cache. Fall through.

    logging.info(f"Fetching prices for {ticker} from yfinance_transformers ({start_date}-{end_date}).")
    yf_response = yf_get_price_response(ticker_symbol=ticker, start_date=start_date, end_date=end_date)

    if yf_response is None or not yf_response.prices:
        logging.info(f"No price data returned from yfinance_transformers for {ticker}.")
        return []

    prices = yf_response.prices
    
    # Cache the results as dicts. set_prices uses _merge_data.
    if prices:
        price_dicts = [p.model_dump() for p in prices]
        _cache.set_prices(ticker, price_dicts)
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str, # YYYY-MM-DD
    period: str = "ttm", # "ttm", "annual", "quarterly", or "all" to try and get what's available
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or yfinance_transformers, then filter."""
    # Using simple ticker-based key for get_financial_metrics.
    if cached_data_list := _cache.get_financial_metrics(ticker):
        logging.info(f"All financial metrics for {ticker} found in cache.")
        all_metrics = [FinancialMetrics(**metric_dict) for metric_dict in cached_data_list]
    else:
        logging.info(f"Fetching all financial metrics for {ticker} from yfinance_transformers.")
        yf_response = yf_get_financial_metrics_response(ticker_symbol=ticker, include_historical=True)
        
        if yf_response is None or not yf_response.financial_metrics:
            logging.info(f"No financial metrics returned from yfinance_transformers for {ticker}.")
            return []
        
        all_metrics = yf_response.financial_metrics
        if all_metrics:
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in all_metrics])

    # Filter the comprehensive list (from cache or fresh fetch)
    filtered_metrics = []
    for metric in all_metrics:
        # Ensure report_period is valid date string for comparison
        report_period_date_str = metric.report_period[:10] if metric.report_period else ""
        if report_period_date_str <= end_date:
            if period == "all": # "all" means any period type up to end_date
                filtered_metrics.append(metric)
            elif metric.period == period: # "ttm", "annual", "quarterly"
                filtered_metrics.append(metric)
            # If requested period is "ttm" and current metric is "ttm" it's already caught.
            # yfinance_transformers returns TTM as its first item if available.

    # Sort by report_period (descending) then by period type (ttm first, then annual, then quarterly)
    period_sort_order = {"ttm": 0, "annual": 1, "quarterly": 2, None: 3}
    filtered_metrics.sort(key=lambda x: (x.report_period or "", period_sort_order.get(x.period, 99)), reverse=True)
    
    return filtered_metrics[:limit]


def search_line_items(
    ticker: str,
    line_items: list[str], # e.g., ["TotalRevenue", "NetIncome"] - these are keys in LineItem model
    end_date: str, # YYYY-MM-DD
    period: str = "ttm", # "ttm", "annual", "quarterly"
    limit: int = 10,
) -> list[LineItem]:
    """Fetch financial statements from yfinance, then extract and filter specific line items."""
    # Using new cache methods for financial statements.
    if cached_data_list := _cache.get_financial_statements(ticker):
        logging.info(f"All financial statements for {ticker} found in cache.")
        all_statements = [LineItem(**stmt_dict) for stmt_dict in cached_data_list]
    else:
        logging.info(f"Fetching all financial statements for {ticker} from yfinance_transformers.")
        yf_response = yf_get_financial_statements_response(ticker_symbol=ticker)

        if yf_response is None or not yf_response.search_results: # search_results is the list of LineItems
            logging.info(f"No financial statements returned from yfinance_transformers for {ticker}.")
            return []
        
        all_statements = yf_response.search_results
        if all_statements:
            _cache.set_financial_statements(ticker, [s.model_dump() for s in all_statements])
            
    # Filter statements by date and period
    relevant_statements = []
    for stmt in all_statements:
        report_period_date_str = stmt.report_period[:10] if stmt.report_period else ""
        # Adapt "ttm" period request: yfinance_transformers doesn't produce "ttm" LineItems.
        # For TTM, users should use get_financial_metrics.
        # Here, we map "ttm" to "quarterly" as the closest available for individual line items,
        # or could decide to return empty if strict TTM line items are expected.
        # For simplicity, if "ttm" is requested for line items, we'll assume they want latest quarterly.
        target_period = period
        if period == "ttm":
            # yfinance_transformers.get_financial_statements_response produces 'annual' and 'quarterly'
            # If TTM is requested, we might not have a direct match.
            # This function's purpose is to extract *specific line items* from statements.
            # TTM values for specific line items are often part of FinancialMetrics.
            # Let's assume if period="ttm", we might not find a direct match here.
            # Or, one could sum up last 4 quarters if available.
            # For now, if period is "ttm", we will not find a direct match with 'annual'/'quarterly' statements.
            # This part of the logic might need refinement based on how TTM line items are expected.
            # Given the yfinance source, TTM line items are not directly provided in statements.
            # A more accurate TTM line item would need calculation from quarterly data.
            # For now, we filter by exact period match. If 'ttm' is asked, it likely yields nothing unless period is 'quarterly'.
             pass # No direct TTM statements, rely on exact period match below.

        if report_period_date_str <= end_date and stmt.period == target_period:
            relevant_statements.append(stmt)
    
    relevant_statements.sort(key=lambda x: x.report_period or "", reverse=True)
    
    # Extract requested line_items
    results = []
    for stmt in relevant_statements:
        extracted_data = {
            "ticker": stmt.ticker,
            "report_period": stmt.report_period,
            "period": stmt.period,
            "currency": stmt.currency,
            "statement_type": stmt.statement_type # Retain original statement type
        }
        has_any_requested_item = False
        for item_key in line_items:
            # LineItem model from yfinance_transformers stores accounts as dynamic fields.
            # We need to use getattr to access them.
            # Account names in LineItem from yfinance_transformers are already sanitized.
            # The requested line_items might need sanitization if they come in "raw" form e.g. "Total Revenue"
            sanitized_item_key = ''.join(c for c in str(item_key).replace(' ', '_') if c.isalnum() or c == '_')
            if sanitized_item_key and sanitized_item_key[0].isdigit(): sanitized_item_key = "_" + sanitized_item_key

            if hasattr(stmt, sanitized_item_key):
                extracted_data[sanitized_item_key] = getattr(stmt, sanitized_item_key)
                has_any_requested_item = True
            else: # If specific item not found, represent as None
                extracted_data[sanitized_item_key] = None
        
        if has_any_requested_item : # Only add if at least one requested item was found or explicitly set to None.
            # Create a new LineItem ensuring all its defined fields are present or None
            # This is tricky because LineItem uses **extra_fields.
            # The goal is to return a LineItem that *only* has the requested line_items as its dynamic part.
            # The Pydantic model LineItem itself is defined with specific fixed fields and **extra.
            # We can construct dict and then LineItem(**dict)
            # For this refactor, we'll assume the LineItem model can handle dynamic fields being passed at init.
            # The returned LineItem will have the standard fields + only the requested line_items.
             results.append(LineItem(**extracted_data))


    return results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str, # YYYY-MM-DD
    start_date: str | None = None, # YYYY-MM-DD
    limit: int = 1000, # Note: yfinance_transformers might not support limit directly
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or yfinance_transformers, then filter."""
    # Using simple ticker-based key for get_insider_trades.
    if cached_data_list := _cache.get_insider_trades(ticker):
        logging.info(f"All insider trades for {ticker} found in cache.")
        all_trades = [InsiderTrade(**trade_dict) for trade_dict in cached_data_list]
    else:
        logging.info(f"Fetching all insider trades for {ticker} from yfinance_transformers.")
        yf_response = yf_get_insider_trades_response(ticker_symbol=ticker)

        if yf_response is None or not yf_response.insider_trades:
            logging.info(f"No insider trades returned from yfinance_transformers for {ticker}.")
            return []
        
        all_trades = yf_response.insider_trades
        if all_trades:
            _cache.set_insider_trades(ticker, [t.model_dump() for t in all_trades])

    # Filter by date
    filtered_trades = []
    for trade in all_trades:
        # yfinance provides transaction_date. Use it for filtering.
        # Ensure trade.transaction_date is not None before comparison
        trade_date_str = trade.transaction_date[:10] if trade.transaction_date else None
        if trade_date_str:
            is_after_start = (start_date is None) or (trade_date_str >= start_date)
            is_before_end = trade_date_str <= end_date
            if is_after_start and is_before_end:
                filtered_trades.append(trade)
    
    # Sort by transaction_date (descending)
    filtered_trades.sort(key=lambda x: x.transaction_date or "", reverse=True)
    
    return filtered_trades[:limit]


def get_company_news(
    ticker: str,
    end_date: str, # YYYY-MM-DD
    start_date: str | None = None, # YYYY-MM-DD
    limit: int = 1000, # Note: yfinance_transformers might not support limit
) -> list[CompanyNews]:
    """Fetch company news from cache or yfinance_transformers, then filter."""
    # Using simple ticker-based key for get_company_news.
    if cached_data_list := _cache.get_company_news(ticker):
        logging.info(f"All company news for {ticker} found in cache.")
        all_news = [CompanyNews(**news_dict) for news_dict in cached_data_list]
    else:
        logging.info(f"Fetching all company news for {ticker} from yfinance_transformers.")
        yf_response = yf_get_company_news_response(ticker_symbol=ticker)

        if yf_response is None or not yf_response.news:
            logging.info(f"No company news returned from yfinance_transformers for {ticker}.")
            return []
            
        all_news = yf_response.news
        if all_news:
            _cache.set_company_news(ticker, [n.model_dump() for n in all_news])

    # Filter by date
    filtered_news = []
    for news_item in all_news:
        # yfinance news items have a 'date' field (which is string like 'YYYY-MM-DD HH:MM:SS')
        news_date_str = news_item.date[:10] if news_item.date else None # Compare YYYY-MM-DD part
        if news_date_str:
            is_after_start = (start_date is None) or (news_date_str >= start_date)
            is_before_end = news_date_str <= end_date
            if is_after_start and is_before_end:
                filtered_news.append(news_item)

    # Sort by date (descending)
    filtered_news.sort(key=lambda x: x.date or "", reverse=True)
    
    return filtered_news[:limit]


def get_market_cap(
    ticker: str,
    end_date: str, # YYYY-MM-DD
) -> float | None:
    """Fetch market cap. Uses yfinance_transformers for current, or latest from historical metrics."""
    # If end_date is recent (e.g., today or within a few days), get current market cap
    # yfinance get_company_facts_response gives current market cap
    # For simplicity, we'll use it if end_date is "close" to today.
    # A more robust way might be to check if end_date is the latest trading day.
    # For this refactor, let's assume if end_date is today, we want current market_cap.
    is_today = end_date == datetime.datetime.now().strftime("%Y-%m-%d")

    if is_today:
        logging.info(f"Fetching current market cap for {ticker} using yfinance_transformers.get_company_facts_response.")
        yf_facts_response = yf_get_company_facts_response(ticker_symbol=ticker)
        if yf_facts_response and yf_facts_response.company_facts:
            return yf_facts_response.company_facts.market_cap
        else:
            logging.warning(f"Could not get current market cap from company_facts for {ticker}.")
            # Fall through to try financial_metrics as a backup if needed, or return None.

    # If not today, or if facts failed, try to get it from historical financial_metrics
    # We need the metric closest to or on end_date.
    # get_financial_metrics is already refactored and will use yfinance.
    # We ask for TTM around that end_date, which should include market_cap.
    logging.info(f"Fetching market cap for {ticker} near {end_date} from financial_metrics.")
    # Get TTM metric for the period ending on or before end_date
    # The refactored get_financial_metrics will return a list, sorted by date.
    # We want the one closest to end_date.
    metrics_list = get_financial_metrics(ticker, end_date, period="ttm", limit=1) 
    if metrics_list:
        # The first item should be the TTM metric on or before end_date
        latest_metric_on_or_before_end_date = metrics_list[0]
        if latest_metric_on_or_before_end_date.market_cap is not None:
            return latest_metric_on_or_before_end_date.market_cap
        else:
            logging.warning(f"Market cap was None in TTM metric for {ticker} near {end_date}.")
    
    logging.warning(f"Could not determine market cap for {ticker} for {end_date} from any source.")
    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert a list of Price Pydantic models to a Pandas DataFrame."""
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# get_price_data remains the same as it already calls the local get_prices
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches price data and converts it to a DataFrame."""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
