import uuid
from datetime import datetime, date
from typing import List, Optional
from sqlalchemy import (
    String, Integer, Float, Text, Date, DateTime, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()

class Company(Base):
    """
    Stores corporate entity master data.
    """
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    cik: Mapped[Optional[str]] = mapped_column(String(20), index=True, nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="company", cascade="all, delete-orphan")
    metrics: Mapped[List["FinancialMetric"]] = relationship("FinancialMetric", back_populates="company", cascade="all, delete-orphan")


class Document(Base):
    """
    Tracks ingested SEC filings (10-K, 10-Q) with SHA-256 deduplication and AI summaries.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    
    # Document Metadata
    form_type: Mapped[str] = mapped_column(String(20), nullable=False) # e.g. '10-K', '10-Q'
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    fiscal_period: Mapped[str] = mapped_column(String(20), nullable=False) # e.g. 'FY2024', 'Q3'
    filing_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Ingestion & Cost Optimization Controls
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # One-Time Cached AI Synthesis (Free to read forever after ingestion)
    executive_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_risks: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    growth_catalysts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    company: Mapped["Company"] = relationship("Company", back_populates="documents")
    metrics: Mapped[List["FinancialMetric"]] = relationship("FinancialMetric", back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class FinancialMetric(Base):
    """
    Stores fundamental financial statement line items and calculated accounting ratios.
    """
    __tablename__ = "financial_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)

    # Core Income Statement Metrics (in millions/billions USD)
    revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_income: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_income: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    diluted_eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cash Flow & Balance Sheet Metrics
    operating_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    capital_expenditures: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    free_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cash_and_equivalents: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_debt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shareholders_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Precalculated Financial Ratios
    gross_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_profit_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    company: Mapped["Company"] = relationship("Company", back_populates="metrics")
    document: Mapped["Document"] = relationship("Document", back_populates="metrics")


class DocumentChunk(Base):
    """
    Stores section-aware text & table chunks with pgvector embeddings for hybrid retrieval.
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    
    # Financial Structural Context
    section: Mapped[str] = mapped_column(String(100), index=True, nullable=False) # e.g. 'Item 7. MD&A', 'Item 8. Financial Statements'
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(20), default="text") # 'text' or 'table'
    
    # Chunk Content & Vector Embedding
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    # Composite index for ultra-fast filtered vector retrieval
    __table_args__ = (
        Index("ix_chunks_ticker_section", "ticker", "section"),
    )