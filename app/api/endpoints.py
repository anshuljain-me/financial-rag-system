from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pathlib import Path

from app.core.security import verify_api_key
from app.core.sanitizer import FinancialQuerySanitizer
from app.core.limiter import rate_limiter
from app.core.database import AsyncSessionLocal
from app.models.domain import Company, Document, FinancialMetric
from app.rag.qa_engine import FinancialQAService
from app.analytics.technical import TechnicalAnalysisEngine
from app.ingestion.pipeline import IngestionPipeline
from sqlalchemy import select

router = APIRouter()
pipeline = IngestionPipeline()
qa_service = FinancialQAService()
ta_engine = TechnicalAnalysisEngine()

class ChatQueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = "ALL"

@router.post("/chat")
async def chat_query(
    req: ChatQueryRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Authenticated & Rate-Limited Financial RAG Query Endpoint.
    """
    rate_limiter.check_rate_limit(request, max_requests=30, window_seconds=60)
    
    clean_q, is_flagged = FinancialQuerySanitizer.sanitize_query(req.question)
    clean_ticker = FinancialQuerySanitizer.validate_ticker(req.ticker)

    if not clean_q:
        raise HTTPException(status_code=400, detail="Invalid or empty query string.")

    response = await qa_service.answer_question(question=clean_q, ticker=clean_ticker)
    response["query_sanitized"] = is_flagged
    return response

@router.get("/companies")
async def list_companies(
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    rate_limiter.check_rate_limit(request, max_requests=60, window_seconds=60)
    async with AsyncSessionLocal() as session:
        stmt = select(Company).order_by(Company.ticker)
        res = await session.execute(stmt)
        companies = res.scalars().all()
        return [{"id": str(c.id), "ticker": c.ticker, "name": c.company_name} for c in companies]

@router.get("/company/{ticker}/technical")
async def get_technical_analysis(
    ticker: str,
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    rate_limiter.check_rate_limit(request, max_requests=30, window_seconds=60)
    clean_t = FinancialQuerySanitizer.validate_ticker(ticker)
    return ta_engine.fetch_and_analyze(clean_t)
