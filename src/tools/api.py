import datetime
# import os # No longer needed
import pandas as pd
# import requests # No longer needed
import yfinance as yf # Ensure yfinance is imported

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or yfinance."""
    cache_key = f"prices_{ticker}_{start_date}_{end_date}"

    if cached_price_data_dicts := _cache.get_prices(cache_key): # Use specific key
        # Data is already for the specific date range due to the key
        # Ensure volume is int for cached data
        reconstructed_prices = []
        for price_dict in cached_price_data_dicts:
            price_obj = Price(**price_dict)
            price_obj.volume = int(price_obj.volume)
            reconstructed_prices.append(price_obj)
        return reconstructed_prices

    # If not in cache, fetch from yfinance
    try:
        yf_ticker_obj = yf.Ticker(ticker)
        # yfinance end date for history is exclusive, so add one day to include the end_date.
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        adjusted_end_date_str = (end_date_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        history_df = yf_ticker_obj.history(start=start_date, end=adjusted_end_date_str, interval="1d")

        if history_df.empty:
            _cache.set_prices(cache_key, []) # Cache empty result for this specific key
            return []

        prices_to_model = []
        for date_val, row_data in history_df.iterrows():
            time_str = date_val.strftime("%Y-%m-%d")

            # Filter to ensure data is strictly within the original requested start_date and end_date inclusive
            if not (start_date <= time_str <= end_date):
                continue

            price_obj = Price(
                time=time_str,
                open=row_data["Open"],
                high=row_data["High"],
                low=row_data["Low"],
                close=row_data["Close"],
                volume=int(row_data["Volume"])
            )
            prices_to_model.append(price_obj)

        if not prices_to_model:
            _cache.set_prices(cache_key, []) # Cache empty if no data after filtering
            return []
        
        _cache.set_prices(cache_key, [p.model_dump() for p in prices_to_model])
        return prices_to_model

    except Exception as e:
        print(f"Error fetching price data for {ticker} using yfinance: {e}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or yfinance."""
    cache_key = f"financial_metrics_{ticker}_{period}_{end_date}_{limit}" # More specific cache key
    
    if cached_data_list := _cache.get_financial_metrics(cache_key):
        # Data is stored as list of dicts, convert back to FinancialMetrics objects
        # No further filtering by end_date or limit needed here as cache_key includes them
        return [FinancialMetrics(**metric_dict) for metric_dict in cached_data_list]

    yf_ticker = yf.Ticker(ticker)
    all_metrics_data = []

    # Helper to safely get values from yfinance data (dict or Series)
    def _safe_get(data_source, key, default=None):
        if data_source is None:
            return default
        if isinstance(data_source, pd.Series):
            return data_source.get(key, default)
        return data_source.get(key, default)

    try:
        info = yf_ticker.info
        currency = _safe_get(info, "currency", "USD")

        if period == "ttm":
            # For TTM, primarily use .info. 'limit' implies latest TTM data.
            # report_period for TTM is considered as the end_date.
            
            fcf_ttm = None
            # Try to get TTM Free Cash Flow by summing the last 4 quarters
            try:
                q_cashflow = yf_ticker.quarterly_cashflow
                if not q_cashflow.empty and len(q_cashflow.columns) >= 4:
                    if "Free Cash Flow" in q_cashflow.index: # yfinance often has this pre-calculated
                         fcf_ttm = q_cashflow.loc["Free Cash Flow"].iloc[:4].sum()
                    elif "Total Cash From Operating Activities" in q_cashflow.index and "Capital Expenditures" in q_cashflow.index:
                        op_cash_ttm = q_cashflow.loc["Total Cash From Operating Activities"].iloc[:4].sum()
                        cap_ex_ttm = q_cashflow.loc["Capital Expenditures"].iloc[:4].sum()
                        fcf_ttm = op_cash_ttm + cap_ex_ttm # CapEx is usually negative
                    else: # try another common naming
                        op_cash_ttm = _safe_get(q_cashflow.loc["Operating Cash Flow",:].iloc[:4].sum())
                        cap_ex_ttm = _safe_get(q_cashflow.loc["Capital Expenditure",:].iloc[:4].sum())
                        if op_cash_ttm is not None and cap_ex_ttm is not None:
                           fcf_ttm = op_cash_ttm + cap_ex_ttm


            except Exception: # Broad except as yfinance calls can fail
                fcf_ttm = None
            
            market_cap = _safe_get(info, "marketCap")
            shares_outstanding = _safe_get(info, "sharesOutstanding")

            metrics = FinancialMetrics(
                ticker=ticker,
                report_period=end_date,
                period="ttm",
                currency=currency,
                market_cap=market_cap,
                enterprise_value=_safe_get(info, "enterpriseValue"),
                price_to_earnings_ratio=_safe_get(info, "trailingPE"),
                price_to_book_ratio=_safe_get(info, "priceToBook"),
                price_to_sales_ratio=_safe_get(info, "priceToSalesTrailing12Months"),
                enterprise_value_to_ebitda_ratio=_safe_get(info, "enterpriseToEbitda"),
                enterprise_value_to_revenue_ratio=_safe_get(info, "enterpriseToRevenue"),
                free_cash_flow_yield=(fcf_ttm / market_cap) if fcf_ttm and market_cap else None,
                peg_ratio=_safe_get(info, "pegRatio"),
                gross_margin=_safe_get(info, "grossMargins"),
                operating_margin=_safe_get(info, "operatingMargins"),
                net_margin=_safe_get(info, "profitMargins"),
                return_on_equity=_safe_get(info, "returnOnEquity"),
                return_on_assets=_safe_get(info, "returnOnAssets"),
                payout_ratio=_safe_get(info, "payoutRatio"),
                earnings_per_share=_safe_get(info, "trailingEps"),
                revenue_growth=_safe_get(info, "revenueGrowth"), # Often yoy quarterly from .info
                earnings_growth=_safe_get(info, "earningsQuarterlyGrowth"), # from .info
                current_ratio=_safe_get(info, "currentRatio"),
                quick_ratio=_safe_get(info, "quickRatio"),
                cash_ratio=_safe_get(info, "cashRatio"),
                debt_to_equity=_safe_get(info, "debtToEquity"),
                book_value_per_share=_safe_get(info, "bookValue"), # This is per share in .info
                free_cash_flow_per_share=(fcf_ttm / shares_outstanding) if fcf_ttm and shares_outstanding else None,
                # Fields below are typically harder to get reliable TTM for from .info alone or require more involved calcs
                return_on_invested_capital=None, asset_turnover=None, inventory_turnover=None, receivables_turnover=None,
                days_sales_outstanding=None, operating_cycle=None, working_capital_turnover=None,
                operating_cash_flow_ratio=None, debt_to_assets=None, interest_coverage=None,
                book_value_growth=None, earnings_per_share_growth=None, free_cash_flow_growth=None,
                operating_income_growth=None, ebitda_growth=None
            )
            all_metrics_data.append(metrics)

        elif period in ["annual", "quarterly"]:
            stmt_map = {
                "annual": {
                    "financials": yf_ticker.financials,
                    "balance_sheet": yf_ticker.balance_sheet,
                    "cashflow": yf_ticker.cashflow,
                },
                "quarterly": {
                    "financials": yf_ticker.quarterly_financials,
                    "balance_sheet": yf_ticker.quarterly_balance_sheet,
                    "cashflow": yf_ticker.quarterly_cashflow,
                },
            }

            fin_stmt = stmt_map[period]["financials"]
            bs_stmt = stmt_map[period]["balance_sheet"]
            cf_stmt = stmt_map[period]["cashflow"]

            if fin_stmt.empty:
                return []

            fin_stmt_T = fin_stmt.T.sort_index(ascending=False)
            bs_stmt_T = bs_stmt.T.sort_index(ascending=False)
            cf_stmt_T = cf_stmt.T.sort_index(ascending=False)
            
            # Filter by end_date and apply limit
            valid_indices = [idx for idx in fin_stmt_T.index if idx.strftime("%Y-%m-%d") <= end_date]
            fin_stmt_T = fin_stmt_T.loc[valid_indices].head(limit)

            prev_fin_row = None # For growth calculations

            for report_date_ts, fin_row in fin_stmt_T.iterrows():
                report_date_str = report_date_ts.strftime("%Y-%m-%d")
                
                bs_row = bs_stmt_T[bs_stmt_T.index == report_date_ts].iloc[0] if report_date_ts in bs_stmt_T.index else pd.Series(dtype='float64')
                cf_row = cf_stmt_T[cf_stmt_T.index == report_date_ts].iloc[0] if report_date_ts in cf_stmt_T.index else pd.Series(dtype='float64')

                # Basic items
                total_revenue = _safe_get(fin_row, "Total Revenue", _safe_get(fin_row, "Revenue"))
                net_income = _safe_get(fin_row, "Net Income", _safe_get(fin_row, "NetIncome"))
                gross_profit = _safe_get(fin_row, "Gross Profit", _safe_get(fin_row, "GrossProfit"))
                operating_income = _safe_get(fin_row, "Operating Income", _safe_get(fin_row, "EBIT", _safe_get(fin_row, "OperatingIncome"))) # EBIT can be a proxy
                ebitda = _safe_get(fin_row, "EBITDA")


                total_assets = _safe_get(bs_row, "Total Assets", _safe_get(bs_row, "TotalAssets"))
                total_liabilities = _safe_get(bs_row, "Total Liab", _safe_get(bs_row, "TotalLiabilities"))
                total_equity = _safe_get(bs_row, "Total Stockholder Equity", _safe_get(bs_row, "StockholdersEquity"))
                current_assets = _safe_get(bs_row, "Total Current Assets", _safe_get(bs_row, "CurrentAssets"))
                current_liabilities = _safe_get(bs_row, "Total Current Liabilities", _safe_get(bs_row, "CurrentLiabilities"))
                cash_and_equivalents = _safe_get(bs_row, "Cash And Cash Equivalents", _safe_get(bs_row, "Cash"))
                inventory = _safe_get(bs_row, "Inventory")
                
                operating_cash_flow = _safe_get(cf_row, "Total Cash From Operating Activities", _safe_get(cf_row, "OperatingCashFlow"))
                capital_expenditures = _safe_get(cf_row, "Capital Expenditures", _safe_get(cf_row, "CapitalExpenditure"))
                free_cash_flow = (_safe_get(cf_row, "Free Cash Flow") if "Free Cash Flow" in cf_row else 
                                  (operating_cash_flow + capital_expenditures if operating_cash_flow and capital_expenditures else None))


                # Ratios
                gross_margin = (gross_profit / total_revenue) if gross_profit and total_revenue else None
                operating_margin = (operating_income / total_revenue) if operating_income and total_revenue else None
                net_margin = (net_income / total_revenue) if net_income and total_revenue else None
                
                return_on_equity = (net_income / total_equity) if net_income and total_equity else None # Simplified ROE
                return_on_assets = (net_income / total_assets) if net_income and total_assets else None # Simplified ROA
                
                current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities else None
                quick_ratio = ((current_assets - inventory) / current_liabilities) if current_assets and inventory and current_liabilities else None
                cash_ratio = (cash_and_equivalents / current_liabilities) if cash_and_equivalents and current_liabilities else None
                
                debt_to_equity = (total_liabilities / total_equity) if total_liabilities and total_equity else None # Using total liabilities as proxy for total debt
                debt_to_assets = (total_liabilities / total_assets) if total_liabilities and total_assets else None

                # Per share items
                basic_eps = _safe_get(fin_row, "Basic EPS") # yfinance often provides this
                # Diluted EPS also available: _safe_get(fin_row, "Diluted EPS")
                
                # Shares outstanding can be inferred if not directly available, e.g. Net Income / Basic EPS
                # Or from yf_ticker.info['sharesOutstanding'] for current, but historical is tricky.
                # yfinance statements don't usually list historical shares directly.
                # For simplicity, we'll leave historical book_value_per_share and fcf_per_share as None if shares aren't in statement.
                # Some statements might have "Basic Average Shares" or "Diluted Average Shares"
                shares_stmt = _safe_get(fin_row, "Basic Average Shares", _safe_get(fin_row, "Diluted Average Shares"))

                book_value_per_share = (total_equity / shares_stmt) if total_equity and shares_stmt else None
                free_cash_flow_per_share = (free_cash_flow / shares_stmt) if free_cash_flow and shares_stmt else None

                # Growth rates (simple YOY or QOQ if prev_fin_row exists)
                revenue_growth = None
                earnings_growth = None
                if prev_fin_row is not None:
                    prev_total_revenue = _safe_get(prev_fin_row, "Total Revenue", _safe_get(prev_fin_row, "Revenue"))
                    prev_net_income = _safe_get(prev_fin_row, "Net Income", _safe_get(prev_fin_row, "NetIncome"))
                    if total_revenue and prev_total_revenue:
                        revenue_growth = (total_revenue - prev_total_revenue) / abs(prev_total_revenue)
                    if net_income and prev_net_income:
                        # Avoid division by zero if prev_net_income is 0
                        earnings_growth = (net_income - prev_net_income) / abs(prev_net_income) if prev_net_income else (float('inf') if net_income > 0 else (float('-inf') if net_income < 0 else 0))


                metrics_item = FinancialMetrics(
                    ticker=ticker, report_period=report_date_str, period=period, currency=currency,
                    market_cap=None, enterprise_value=None, # These are point-in-time, hard for historical reports
                    price_to_earnings_ratio=None, price_to_book_ratio=None, price_to_sales_ratio=None, # Need historical price
                    enterprise_value_to_ebitda_ratio=None, enterprise_value_to_revenue_ratio=None, free_cash_flow_yield=None, peg_ratio=None,
                    gross_margin=gross_margin, operating_margin=operating_margin, net_margin=net_margin,
                    return_on_equity=return_on_equity, return_on_assets=return_on_assets,
                    payout_ratio=_safe_get(fin_row, "Payout Ratio"), # Sometimes available
                    earnings_per_share=basic_eps,
                    current_ratio=current_ratio, quick_ratio=quick_ratio, cash_ratio=cash_ratio,
                    debt_to_equity=debt_to_equity, debt_to_assets=debt_to_assets,
                    book_value_per_share=book_value_per_share, free_cash_flow_per_share=free_cash_flow_per_share,
                    revenue_growth=revenue_growth, earnings_growth=earnings_growth,
                    # Remaining fields default to None as they are more complex or less commonly available directly
                    return_on_invested_capital=None, asset_turnover=None, inventory_turnover=None, receivables_turnover=None,
                    days_sales_outstanding=None, operating_cycle=None, working_capital_turnover=None,
                    operating_cash_flow_ratio=None, interest_coverage=None, book_value_growth=None,
                    earnings_per_share_growth=None, free_cash_flow_growth=None, operating_income_growth=None, ebitda_growth=None
                    # Removed duplicate and semantically incorrect: ebitda_growth=ebitda 
                )
                all_metrics_data.append(metrics_item)
                prev_fin_row = fin_row # Update prev_row for next iteration's growth calc
        else:
            raise ValueError(f"Unsupported period: {period}. Must be 'ttm', 'annual', or 'quarterly'.")

        if not all_metrics_data:
            return []

        # Cache the results as a list of dicts
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in all_metrics_data])
        return all_metrics_data

    except Exception as e:
        print(f"Error fetching or processing financial metrics for {ticker} using yfinance: {e}")
        # Attempt to clear potentially corrupted cache entry on error
        try:
            _cache.delete_financial_metrics(cache_key)
        except Exception as cache_e:
            print(f"Error clearing cache for {cache_key}: {cache_e}")
        return []


def search_line_items(
    ticker: str,
    line_items_requested: list[str], 
    end_date: str,
    period: str = "ttm", # 'ttm', 'annual', 'quarterly'
    limit: int = 10,
) -> list[LineItem]:
    """Fetch specific line items from yfinance financial statements, with caching."""
    
    # Generate a cache key that includes all parameters that define the request
    sorted_line_items_str = "_".join(sorted(line_items_requested))
    cache_key = f"line_items_{ticker}_{sorted_line_items_str}_{period}_{end_date}_{limit}"

    if cached_data_list := _cache.get_line_items(cache_key):
        # Data is stored as dicts, reconstruct to LineItem objects
        # No further filtering needed as cache key is specific
        return [LineItem(**item_dict) for item_dict in cached_data_list]

    results: list[LineItem] = []
    yf_ticker = yf.Ticker(ticker)

    try:
        ticker_info = yf_ticker.info
        currency = ticker_info.get("currency", "USD")
    except Exception:
        currency = "USD"

    LINE_ITEM_MAP = {
        "Total Revenue": ("Total Revenue", "income", "financials", "quarterly_financials"),
        "Revenue": ("Total Revenue", "income", "financials", "quarterly_financials"),
        "Gross Profit": ("Gross Profit", "income", "financials", "quarterly_financials"),
        "Operating Income": ("Operating Income", "income", "financials", "quarterly_financials"),
        "EBIT": ("EBIT", "income", "financials", "quarterly_financials"),
        "Net Income": ("Net Income", "income", "financials", "quarterly_financials"),
        "Net Income Common Stockholders": ("Net Income From Continuing Operations", "income", "financials", "quarterly_financials"),
        "Basic EPS": ("Basic EPS", "income", "financials", "quarterly_financials"),
        "Diluted EPS": ("Diluted EPS", "income", "financials", "quarterly_financials"),
        "EBITDA": ("EBITDA", "income", "financials", "quarterly_financials"),
        "Total Assets": ("Total Assets", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Total Current Assets": ("Total Current Assets", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Total Liabilities": ("Total Liab", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Total Current Liabilities": ("Total Current Liabilities", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Total Stockholder Equity": ("Total Stockholder Equity", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Cash": ("Cash", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Cash And Cash Equivalents": ("Cash And Cash Equivalents", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Inventory": ("Inventory", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Net Receivables": ("Net Receivables", "balance", "balance_sheet", "quarterly_balance_sheet"),
        "Operating Cash Flow": ("Total Cash From Operating Activities", "cashflow", "cashflow", "quarterly_cashflow"),
        "Total Cash From Operating Activities": ("Total Cash From Operating Activities", "cashflow", "cashflow", "quarterly_cashflow"),
        "Capital Expenditures": ("Capital Expenditures", "cashflow", "cashflow", "quarterly_cashflow"),
        "Free Cash Flow": ("Free Cash Flow", "cashflow", "cashflow", "quarterly_cashflow"),
    }

    for requested_name in line_items_requested:
        if requested_name not in LINE_ITEM_MAP:
            print(f"Warning: Line item '{requested_name}' for ticker {ticker} not recognized. Skipping.")
            continue

        yfinance_name, statement_type, annual_df_key, quarterly_df_key = LINE_ITEM_MAP[requested_name]

        try:
            if period == "ttm":
                quarterly_df_source = getattr(yf_ticker, quarterly_df_key, None)
                if quarterly_df_source is None or quarterly_df_source.empty:
                    print(f"Warning: Quarterly data for '{requested_name}' (ticker {ticker}) not available for TTM. Skipping.")
                    continue
                
                if yfinance_name not in quarterly_df_source.index:
                    print(f"Warning: Line item '{yfinance_name}' (for '{requested_name}', ticker {ticker}) not in quarterly statement {quarterly_df_key}. Skipping.")
                    continue

                df_T = quarterly_df_source.T
                relevant_periods_df = df_T[df_T.index <= pd.to_datetime(end_date)].sort_index(ascending=False)

                if relevant_periods_df.empty:
                    continue

                if statement_type == "balance":
                    latest_period_data = relevant_periods_df.iloc[0]
                    report_date_str = latest_period_data.name.strftime("%Y-%m-%d")
                    value = latest_period_data.get(yfinance_name)
                    if pd.isna(value): value = None
                    results.append(LineItem(ticker=ticker, report_period=report_date_str, period_type="ttm_latest_quarter", line_item_name=requested_name, value=value, currency=currency))
                
                elif statement_type in ["income", "cashflow"]:
                    if len(relevant_periods_df) >= 4:
                        ttm_periods_df = relevant_periods_df.head(4)
                        value = ttm_periods_df[yfinance_name].sum(skipna=False)
                        if pd.isna(value): value = None
                        report_date_str = ttm_periods_df.index[0].strftime("%Y-%m-%d")
                        results.append(LineItem(ticker=ticker, report_period=report_date_str, period_type="ttm", line_item_name=requested_name, value=value, currency=currency))
                    else:
                        print(f"Warning: Not enough quarterly data for TTM sum of '{requested_name}' (ticker {ticker}). Skipping.")

            elif period in ["annual", "quarterly"]:
                df_key = annual_df_key if period == "annual" else quarterly_df_key
                statement_df = getattr(yf_ticker, df_key, None)

                if statement_df is None or statement_df.empty:
                    print(f"Warning: {period.capitalize()} data for '{df_key}' (ticker {ticker}) not available. Skipping '{requested_name}'.")
                    continue
                
                if yfinance_name not in statement_df.index:
                    print(f"Warning: Line item '{yfinance_name}' (for '{requested_name}', ticker {ticker}) not in {period} statement {df_key}. Skipping.")
                    continue

                df_T = statement_df.T
                relevant_periods_df = df_T[df_T.index <= pd.to_datetime(end_date)].sort_index(ascending=False).head(limit)

                for report_date_ts, row_data in relevant_periods_df.iterrows():
                    report_date_str = report_date_ts.strftime("%Y-%m-%d")
                    value = row_data.get(yfinance_name)
                    if pd.isna(value): value = None
                    results.append(LineItem(ticker=ticker, report_period=report_date_str, period_type=period, line_item_name=requested_name, value=value, currency=currency))
            else:
                # This path should ideally not be reached if period is validated upfront or handled by default.
                print(f"Warning: Unsupported period '{period}' for line item '{requested_name}' (ticker {ticker}). Skipping.")
        
        except Exception as e_item:
            print(f"Error processing line item '{requested_name}' for ticker {ticker}: {e_item}")
            # Continue to next line item
            
    if results: # Only cache if there are some results
        _cache.set_line_items(cache_key, [item.model_dump() for item in results])
    elif not (cached_data_list is None): # If there was no cached data, and we still have no results, cache empty list.
        _cache.set_line_items(cache_key, [])


    return results


def get_insider_trades(
    ticker: str,
    end_date: str, # Inclusive end date for transactions
    start_date: str | None = None, # Inclusive start date for transactions
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or yfinance."""
    cache_key = f"insider_trades_{ticker}" # Cache all trades for a ticker

    if cached_data_list := _cache.get_insider_trades(cache_key):
        # Data is stored as list of dicts, convert back to InsiderTrade objects
        all_trades_from_cache = [InsiderTrade(**trade_dict) for trade_dict in cached_data_list]
        # Apply filtering and limit after retrieving from cache
        filtered_trades = []
        for trade in all_trades_from_cache:
            trade_date_str = trade.transaction_date
            if trade_date_str:
                is_after_start = not start_date or trade_date_str >= start_date
                is_before_end = trade_date_str <= end_date
                if is_after_start and is_before_end:
                    filtered_trades.append(trade)
        
        # Sort by transaction date descending before applying limit
        filtered_trades.sort(key=lambda x: x.transaction_date if x.transaction_date else "", reverse=True)
        return filtered_trades[:limit]

    yf_ticker = yf.Ticker(ticker)
    processed_trades: list[InsiderTrade] = []

    try:
        # Get company name and security type from ticker.info
        company_info = yf_ticker.info
        company_name = company_info.get("shortName", company_info.get("longName"))
        security_type = company_info.get("quoteType")
        security_title_map = {
            "EQUITY": "Common Stock",
            "PREFERRED": "Preferred Stock",
            # Add other mappings if needed
        }
        default_security_title = "Stock"
        security_title = security_title_map.get(security_type, default_security_title) if security_type else default_security_title


        insider_tx_df = yf_ticker.insider_transactions
        if insider_tx_df is None or insider_tx_df.empty:
            print(f"No insider transaction data found for {ticker} via yfinance.")
            # Cache empty result to avoid re-fetching for a while (optional, based on cache policy)
            _cache.set_insider_trades(cache_key, [])
            return []

        # yfinance insider_transactions columns: Shares, Value, URL, Text, Insider, Position, Transaction, Start Date, Ownership
        for _, row in insider_tx_df.iterrows():
            transaction_date_dt = row.get("Start Date")
            transaction_date_str = transaction_date_dt.strftime("%Y-%m-%d") if pd.notnull(transaction_date_dt) else None
            
            if not transaction_date_str: # Skip if no valid transaction date
                continue

            insider_title_str = row.get("Position", "")
            is_director = "director" in insider_title_str.lower() if insider_title_str else False
            
            shares = row.get("Shares")
            value = row.get("Value")
            price_per_share = None
            if pd.notnull(value) and pd.notnull(shares) and shares != 0:
                price_per_share = value / shares
            
            # As per model, filing_date uses transaction_date
            filing_date_str = transaction_date_str

            trade = InsiderTrade(
                ticker=ticker,
                company_name=company_name,
                insider_name=row.get("Insider"),
                insider_title=insider_title_str,
                is_board_director=is_director,
                transaction_date=transaction_date_str,
                transaction_type=row.get("Transaction"),
                transaction_shares=float(shares) if pd.notnull(shares) else None,
                transaction_price_per_share=price_per_share,
                transaction_value=float(value) if pd.notnull(value) else None,
                shares_owned_after_transaction=None, # Not available in this yfinance table
                security_title=security_title,
                filing_date=filing_date_str, # Using transaction_date as per model decision
                ownership_type=row.get("Ownership"),
                url_to_filing=row.get("URL"),
                transaction_description=row.get("Text")
            )
            processed_trades.append(trade)
        
        # Cache all processed trades for the ticker before filtering
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in processed_trades])

        # Now apply filtering and limit, similar to cache retrieval logic
        final_filtered_trades = []
        for trade in processed_trades:
            trade_date_str = trade.transaction_date # Already a string
            if trade_date_str: # Should always be true now
                is_after_start = not start_date or trade_date_str >= start_date
                is_before_end = trade_date_str <= end_date
                if is_after_start and is_before_end:
                    final_filtered_trades.append(trade)
        
        # Sort by transaction date descending
        final_filtered_trades.sort(key=lambda x: x.transaction_date if x.transaction_date else "", reverse=True)
        return final_filtered_trades[:limit]

    except Exception as e:
        print(f"Error fetching or processing insider trades for {ticker} using yfinance: {e}")
        # Optionally, clear cache if processing failed significantly
        # _cache.delete_insider_trades(cache_key) # If desired
        return []


def get_company_news(
    ticker: str,
    end_date: str, # Inclusive end date for news
    start_date: str | None = None, # Inclusive start date for news
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or yfinance."""
    cache_key = f"company_news_{ticker}" # Cache all news for a ticker

    if cached_data_list := _cache.get_company_news(cache_key):
        all_news_from_cache = [CompanyNews(**news_dict) for news_dict in cached_data_list]
        # Apply filtering and limit after retrieving from cache
        filtered_news = []
        for news_item in all_news_from_cache:
            news_date_str = news_item.date # Assuming date is already YYYY-MM-DD string
            if news_date_str:
                is_after_start = not start_date or news_date_str >= start_date
                is_before_end = news_date_str <= end_date
                if is_after_start and is_before_end:
                    filtered_news.append(news_item)
        
        # Sort by date descending before applying limit
        filtered_news.sort(key=lambda x: x.date, reverse=True)
        return filtered_news[:limit]

    yf_ticker = yf.Ticker(ticker)
    processed_news_items: list[CompanyNews] = []

    try:
        raw_news_list = yf_ticker.news
        if not raw_news_list:
            print(f"No news data found for {ticker} via yfinance.")
            _cache.set_company_news(cache_key, []) # Cache empty result
            return []

        for article_dict in raw_news_list:
            content = article_dict.get("content", {})
            if not content: # If content is empty, skip
                content = article_dict # Fallback to top-level if 'content' field is missing (less common)


            uuid = article_dict.get("id", article_dict.get("uuid")) # Prefer 'id' from exploration
            title = content.get("title")
            
            pub_date_raw = content.get("pubDate") # Explored: 'pubDate' under 'content'
            if not pub_date_raw: # Fallback for older yfinance versions or different structures
                 pub_date_raw = article_dict.get("providerPublishTime") # This is often a timestamp

            news_date_str = None
            if pub_date_raw:
                try:
                    # If it's a number (timestamp)
                    if isinstance(pub_date_raw, (int, float)):
                        news_date_str = pd.to_datetime(pub_date_raw, unit='s').strftime('%Y-%m-%d')
                    # If it's a string (like ISO 8601)
                    else:
                        news_date_str = pd.to_datetime(pub_date_raw).strftime('%Y-%m-%d')
                except Exception as e_date:
                    print(f"Could not parse date '{pub_date_raw}' for news item {uuid}: {e_date}")
                    continue # Skip if date cannot be parsed

            if not title or not uuid or not news_date_str: # Essential fields
                continue

            provider = content.get("provider", {})
            source_name = provider.get("displayName")

            url_data = content.get("canonicalUrl", content.get("clickThroughUrl", {})) # Prefer canonicalUrl
            url = url_data.get("url") if isinstance(url_data, dict) else None
            if not url: # Fallback for simpler link structure
                url = article_dict.get("link")


            image_url = None
            thumbnail_data = content.get("thumbnail")
            if isinstance(thumbnail_data, dict):
                resolutions = thumbnail_data.get("resolutions")
                if isinstance(resolutions, list) and resolutions:
                    # Prefer original or highest resolution, fallback to first available
                    image_url = next((res.get("url") for res in resolutions if res.get("tag") == "original"), None)
                    if not image_url:
                        image_url = resolutions[-1].get("url") # Last one often highest listed
                    if not image_url:
                         image_url = resolutions[0].get("url") # First available


            description = content.get("summary", content.get("description"))

            news_item = CompanyNews(
                uuid=uuid,
                ticker=ticker,
                title=title,
                date=news_date_str,
                source_name=source_name,
                url=url,
                image_url=image_url,
                description=description,
                tickers_mentioned=[ticker], # As news is fetched for a specific ticker
                sentiment=None # yfinance does not provide sentiment
            )
            processed_news_items.append(news_item)
        
        # Cache all processed news for the ticker before filtering
        _cache.set_company_news(cache_key, [n.model_dump() for n in processed_news_items])

        # Apply filtering and limit
        final_filtered_news = []
        for news_item_obj in processed_news_items: # Use already created objects
            news_date_str = news_item_obj.date
            # Date filtering (start_date and end_date are YYYY-MM-DD strings)
            is_after_start = not start_date or news_date_str >= start_date
            is_before_end = news_date_str <= end_date
            if is_after_start and is_before_end:
                final_filtered_news.append(news_item_obj)
        
        # Sort by date descending
        final_filtered_news.sort(key=lambda x: x.date, reverse=True)
        return final_filtered_news[:limit]

    except Exception as e:
        print(f"Error fetching or processing company news for {ticker} using yfinance: {e}")
        return []


def get_market_cap(
    ticker: str,
    end_date: str, # Date for which market cap is requested
) -> float | None:
    """
    Fetch current market cap for a ticker using yfinance.
    Note: yfinance primarily provides current market cap. This function will return
    the current market cap regardless of the end_date.
    If end_date is historical, this value may not be accurate for that date.
    """
    
    # Check if end_date is not today and print a warning if so.
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    if end_date != today_str:
        print(f"Warning: get_market_cap for ticker {ticker} called with end_date {end_date}. "
              f"This function returns the CURRENT market cap from yfinance, not historical.")

    try:
        yf_ticker_obj = yf.Ticker(ticker)
        info = yf_ticker_obj.info
        market_cap = info.get("marketCap")

        if market_cap is not None:
            return float(market_cap)
        else:
            print(f"Market cap not found for ticker {ticker} in yfinance info.")
            return None
            
    except Exception as e:
        print(f"Error fetching market cap for {ticker} using yfinance: {e}")
        return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
