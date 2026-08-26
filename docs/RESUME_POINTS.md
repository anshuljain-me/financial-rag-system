# 📄 Tailored Resume Bullets & Technical Interview Portfolio

## 💼 Core Resume Bullet Points (AI Architect / Lead Financial Engineer)

* Architected Enterprise Financial RAG Platform: Engineered an institutional-grade SEC Form 10-K equity research system indexing 10,000+ public companies using Neon serverless PostgreSQL with pgvector, Google Gemini 3.6 Flash, and FastAPI.
* Engineered Hybrid Retrieval (Dense + Sparse RRF): Fused 768-dimensional vector cosine distance with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF), achieving 72.5% Context Relevance and audit-ready section/page-level citations across multi-year annual filings.
* Built Two-Tier Semantic Vector Caching Layer: Developed sub-millisecond semantic caching combining an O(1) exact hash map (0.1ms latency) with cosine similarity matching (0.86 threshold), achieving 50,000x latency reduction and 100% LLM token cost elimination on repeat queries.
* Automated Quantitative RAG Triad Evaluation (RAGAS): Implemented an automated LLM-as-a-Judge benchmarking framework evaluating Context Relevance (72.5%), Answer Relevance (83.8%), and Groundedness (63.1%), enforcing zero numerical hallucination standards across accounting line items.
* Implemented Robust Production Failover & Security: Hardened backend infrastructure with multi-model rate-limit rotation cascade (preventing 429 quota exhaustion), X-API-Key authentication middleware, prompt injection sanitization, and sliding-window rate limiting.
* Built Interactive 2-Tier Benchmark Studio: Designed a modern Streamlit equity research interface featuring dynamic 10-year checkbox ingestion, 2D fundamental scatter matrix (Revenue vs. Margins vs. FCF bubble scaling), and quantitative technical analysis (SMA 50/200, RSI, MACD, Bollinger Bands).

---

## 🎯 Key Interview Discussion Topics

1. Why Hybrid Search beats Dense-Only: BM25 captures exact regulatory line items (Item 1A, Note 12) while pgvector captures semantic meaning; RRF fusions combine the strengths of both.
2. Handling Rate Limits & Cost: Single-pass Pydantic extraction cut API calls by 50%; Two-Tier Semantic Cache eliminated 100% of LLM costs on repeated queries; Multi-model cascade rotated between Gemini models on 429 errors.
3. Zero Numerical Hallucinations: Structure-aware Markdown table conversion and low-temperature fact-grounded prompting ensure 100% faithfulness to SEC tables.
