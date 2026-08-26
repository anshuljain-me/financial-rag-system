from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CompleteFilingExtractionSchema(BaseModel):
    """
    Unified Single-Pass Schema: Extracts both fundamental financial line items 
    and the executive strategic summary in ONE SINGLE LLM CALL (cuts API calls by 50%).
    """
    company_name: str = Field(description="Official name of the corporation")
    ticker: str = Field(description="Stock ticker symbol (e.g. AAPL, MSFT, NVDA)")
    fiscal_year: int = Field(description="Fiscal year of the report")
    fiscal_period: str = Field(description="Fiscal period (e.g. 'FY2025', 'FY2024')")
    
    # Financial Statement Line Items in Millions of USD
    revenue: Optional[float] = Field(None, description="Total Revenue / Net Sales in Millions USD")
    gross_profit: Optional[float] = Field(None, description="Gross Profit in Millions USD")
    operating_income: Optional[float] = Field(None, description="Operating Income / EBIT in Millions USD")
    net_income: Optional[float] = Field(None, description="Net Income attributable to shareholders in Millions USD")
    diluted_eps: Optional[float] = Field(None, description="Diluted Earnings Per Share in USD")
    
    operating_cash_flow: Optional[float] = Field(None, description="Cash Flow from Operations in Millions USD")
    capital_expenditures: Optional[float] = Field(None, description="Capital Expenditures (CapEx) in Millions USD")
    free_cash_flow: Optional[float] = Field(None, description="Free Cash Flow in Millions USD")
    
    total_cash_and_equivalents: Optional[float] = Field(None, description="Cash & Marketable Securities in Millions USD")
    total_debt: Optional[float] = Field(None, description="Short-term plus Long-term Debt in Millions USD")
    shareholders_equity: Optional[float] = Field(None, description="Total Stockholders Equity in Millions USD")

    # Executive Summary & Risk Disclosures
    executive_summary: str = Field(description="A concise 3-paragraph executive summary of annual performance.")
    key_risks: List[str] = Field(description="Top 3 to 5 material business, legal, or market risks.")
    growth_catalysts: List[str] = Field(description="Top 3 to 5 key growth drivers and catalysts.")

class CitationSource(BaseModel):
    ticker: str
    section: str
    page_number: int
    content_snippet: str
    score: float

class RAGChatResponse(BaseModel):
    answer: str
    citations: List[CitationSource]
    ticker: str
    confidence_score: Optional[float] = None
