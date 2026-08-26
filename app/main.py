from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.endpoints import router as api_router
from app.core.config import get_settings

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Institutional Financial RAG Platform",
    description="Enterprise SEC Form 10-K RAG & Quantitative Technical Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "financial-rag-api",
        "environment": "production"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Institutional Financial RAG & SEC Intelligence API.",
        "documentation": "/docs"
    }
