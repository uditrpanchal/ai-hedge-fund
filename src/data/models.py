from pydantic import BaseModel


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]


class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: float | None
    enterprise_value: float | None
    price_to_earnings_ratio: float | None
    price_to_book_ratio: float | None
    price_to_sales_ratio: float | None
    enterprise_value_to_ebitda_ratio: float | None
    enterprise_value_to_revenue_ratio: float | None
    free_cash_flow_yield: float | None
    peg_ratio: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    return_on_equity: float | None
    return_on_assets: float | None
    return_on_invested_capital: float | None
    asset_turnover: float | None
    inventory_turnover: float | None
    receivables_turnover: float | None
    days_sales_outstanding: float | None
    operating_cycle: float | None
    working_capital_turnover: float | None
    current_ratio: float | None
    quick_ratio: float | None
    cash_ratio: float | None
    operating_cash_flow_ratio: float | None
    debt_to_equity: float | None
    debt_to_assets: float | None
    interest_coverage: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    book_value_growth: float | None
    earnings_per_share_growth: float | None
    free_cash_flow_growth: float | None
    operating_income_growth: float | None
    ebitda_growth: float | None
    payout_ratio: float | None
    earnings_per_share: float | None
    book_value_per_share: float | None
    free_cash_flow_per_share: float | None


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str  # Date of the report
    period_type: str  # 'annual', 'quarterly', or 'ttm'
    line_item_name: str  # Name of the specific financial line item
    value: float | None = None
    currency: str  # Currency of the value (e.g., USD)


class LineItemResponse(BaseModel): # This can now be a simple list if the function returns List[LineItem]
    search_results: list[LineItem] 
    # Alternatively, if the function directly returns List[LineItem], 
    # this specific response model might not be strictly needed by the function's direct caller,
    # but could be used by API endpoints. For now, keeping it as is.


class InsiderTrade(BaseModel):
    ticker: str
    company_name: str | None # e.g., from ticker.info['shortName']
    insider_name: str | None # From yfinance 'Insider'
    insider_title: str | None # From yfinance 'Position'
    is_board_director: bool | None # Heuristic: "director" in insider_title.lower()
    transaction_date: str | None # From yfinance 'Start Date' (actual transaction date)
    transaction_type: str | None # From yfinance 'Transaction' (e.g., "Sale", "Purchase")
    transaction_shares: float | None # From yfinance 'Shares'
    # Calculated: transaction_value / transaction_shares. Handle division by zero.
    transaction_price_per_share: float | None 
    transaction_value: float | None # From yfinance 'Value'
    # The following two are often not available in simple yfinance transaction lists
    shares_owned_before_transaction: float | None = None 
    shares_owned_after_transaction: float | None # From yfinance 'Shares Owned Following Transaction' if available, else None
                                                # Note: yfinance.insider_transactions doesn't seem to have this directly.
                                                # yfinance.insider_roster_holders has 'Shares Owned Directly' but it's a current snapshot.
    security_title: str | None = "Common Stock" # Default value, or from ticker.info['quoteType']
    # yfinance 'Start Date' is transaction date. True SEC filing date not in this specific yf table.
    # Using transaction_date for filing_date if it must be non-optional.
    filing_date: str 
    ownership_type: str | None # From yfinance 'Ownership' (e.g., 'D' for Direct, 'I' for Indirect)
    url_to_filing: str | None # From yfinance 'URL'
    transaction_description: str | None # From yfinance 'Text' (e.g., "Sale at price XXX")


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    uuid: str # Unique ID for the news article
    ticker: str # The primary ticker symbol this news is associated with
    title: str
    date: str # Publication date in YYYY-MM-DD format
    source_name: str | None = None # Name of the news source (e.g., "Yahoo Finance")
    url: str | None = None # URL to the full article
    image_url: str | None = None # URL to a thumbnail or relevant image
    description: str | None = None # A summary or snippet of the news article
    tickers_mentioned: list[str] | None = None # Other tickers potentially mentioned or related
    sentiment: str | None = None # Sentiment score/label (not provided by yfinance, will be None)


class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]


class CompanyFacts(BaseModel):
    ticker: str
    name: str
    cik: str | None = None
    industry: str | None = None
    sector: str | None = None
    category: str | None = None
    exchange: str | None = None
    is_active: bool | None = None
    listing_date: str | None = None
    location: str | None = None
    market_cap: float | None = None
    number_of_employees: int | None = None
    sec_filings_url: str | None = None
    sic_code: str | None = None
    sic_industry: str | None = None
    sic_sector: str | None = None
    website_url: str | None = None
    weighted_average_shares: int | None = None


class CompanyFactsResponse(BaseModel):
    company_facts: CompanyFacts


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
