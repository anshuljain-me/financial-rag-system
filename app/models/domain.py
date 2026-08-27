import uuid
from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    documents = relationship("Document", back_populates="company", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String(36), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), index=True, nullable=False)
    form_type = Column(String(20), default="10-K")
    fiscal_year = Column(Integer, nullable=False)
    fiscal_period = Column(String(20), nullable=False)
    file_path = Column(String(500), nullable=True)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)
    executive_summary = Column(Text, nullable=True)
    key_risks = Column(Text, nullable=True)
    growth_catalysts = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    company = relationship("Company", back_populates="documents")
    metrics = relationship("FinancialMetric", back_populates="document", uselist=False, cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class FinancialMetric(Base):
    __tablename__ = "financial_metrics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    revenue = Column(Float, nullable=True)
    gross_profit = Column(Float, nullable=True)
    operating_income = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    diluted_eps = Column(Float, nullable=True)
    operating_cash_flow = Column(Float, nullable=True)
    capital_expenditures = Column(Float, nullable=True)
    free_cash_flow = Column(Float, nullable=True)
    total_cash_and_equivalents = Column(Float, nullable=True)
    total_debt = Column(Float, nullable=True)
    shareholders_equity = Column(Float, nullable=True)
    
    gross_margin = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    net_profit_margin = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)

    document = relationship("Document", back_populates="metrics")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), index=True, nullable=False)
    section = Column(String(100), default="SEC Disclosures")
    chunk_type = Column(String(20), default="text")
    content = Column(Text, nullable=False)
    page_number = Column(Integer, default=1)
    chunk_index = Column(Integer, default=0)
    embedding = Column(Vector(768), nullable=True)

    document = relationship("Document", back_populates="chunks")
