"""
FastAPI Backend — Pakistan Legal AI
Endpoints:
  GET  /                    — health check
  POST /search              — semantic case search
  POST /analyze             — AI-powered legal analysis
  POST /scrape              — trigger court scraping job
  GET  /stats               — DB statistics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embeddings.vector_db import get_db
from api.llm_engine import analyze_query

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Pakistan Legal AI",
    description="Search and analyze SHC, LHC, IHC judgments using AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise DB on startup
_db = None

@app.on_event("startup")
async def startup():
    global _db
    log.info("Initialising vector database...")
    _db = get_db()

    if _db.count() == 0:
        log.info("DB is empty. Waiting for scraping...")

    log.info(f"DB ready with {_db.count()} cases.")

# ── Request / Response Models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    court: Optional[str] = None   # SHC | LHC | IHC | None (all)

class AnalyzeRequest(BaseModel):
    query: str
    top_k: int = 8
    court: Optional[str] = None

class ScrapeRequest(BaseModel):
    keyword: Optional[str] = None
    courts: list[str] = ["SHC", "LHC", "IHC"]
    max_pages: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Pakistan Legal AI", "version": "1.0.0"}


@app.get("/stats")
def stats():
    db = get_db()
    return {
        "total_cases": db.count(),
        "courts_covered": ["SHC", "LHC", "IHC"],
        "model": "all-MiniLM-L6-v2",
        "db_type": "ChromaDB (local)",
    }


@app.post("/search")
def search_cases(req: SearchRequest):
    """Pure semantic search — returns matching cases ranked by similarity."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    db = get_db()
    results = db.search(req.query, top_k=req.top_k, court_filter=req.court)

    return {
        "query":   req.query,
        "total":   len(results),
        "results": results,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Semantic search + AI legal analysis with citations."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    db = get_db()
    cases = db.search(req.query, top_k=req.top_k, court_filter=req.court)

    if not cases:
        return {
            "query":     req.query,
            "analysis":  "No relevant cases found. Try different keywords or scrape more data.",
            "citations": [],
            "cases":     [],
        }

    result = analyze_query(req.query, cases)

    return {
        "query":     req.query,
        "analysis":  result["analysis"],
        "citations": result["citations"],
        "cases":     result["cases_used"],
        "total_searched": db.count(),
    }


@app.post("/scrape")
def trigger_scrape(req: ScrapeRequest, background_tasks: BackgroundTasks):
    """Kick off a background scraping job for the specified courts."""
    background_tasks.add_task(_run_scrape, req.keyword, req.courts, req.max_pages)
    return {
        "status":  "started",
        "message": f"Scraping {req.courts} in background. Results will auto-index when done.",
        "keyword": req.keyword,
    }


def _run_scrape(keyword, courts, max_pages):
    from scrapers.scraper import SHCScraper, LHCScraper, IHCScraper, enrich_cases

    scraper_map = {
        "SHC": SHCScraper,
        "LHC": LHCScraper,
        "IHC": IHCScraper
    }

    db = get_db()
    all_cases = []

    for court in courts:
        if court not in scraper_map:
            continue

        scraper = scraper_map[court]()

        try:
            # 🚀 FULL scraping (not keyword-based anymore)
            cases = scraper.fetch_all(max_pages=max_pages)

            # ⚡ Fetch full text (multi-threaded)
            cases = enrich_cases(scraper, cases)

            all_cases.extend(cases)

            log.info(f"[{court}] Scraped {len(cases)} cases")

        except Exception as e:
            log.error(f"[{court}] Scrape error: {e}")

    if all_cases:
        db._ingest_cases(all_cases)
        log.info(f"Indexed {len(all_cases)} cases. Total: {db.count()}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
