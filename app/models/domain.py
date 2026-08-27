import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class Company(Base):
    __tablename__ = "companies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="company", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    form_type = Column(String(20), nullable=False) # 10-K
    fiscal_year = Column(Integer, nullable=False)
    fiscal_period = Column(String(20), nullable=False) # FY2025
    filing_date = Column(String(50), nullable=True)
    file_path = Column(String(500), nullable=True)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    
    executive_summary = Column(Text, nullable=True)
    key_risks = Column(Text, nullable=True)
    growth_catalysts = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="documents")
    metrics = relationship("FinancialMetric", back_populates="document", uselist=False, cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class FinancialMetric(Base):
    __tablename__ = "financial_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    revenue = Column(Float, nullable=True)
    gross_profit = Column(Float, nullable=True)
    operating_income = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    diluted_eps = Column(Float, nullable=True)
    
    gross_margin = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    net_profit_margin = Column(Float, nullable=True)
    
    free_cash_flow = Column(Float, nullable=True)
    total_debt = Column(Float, nullable=True)
    total_cash_and_equivalents = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="metrics")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    section = Column(String(50), nullable=True)
    chunk_type = Column(String(20), default="text")
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=True)
    embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_doc_chunks_ticker_section", "ticker", "section"),
    )
