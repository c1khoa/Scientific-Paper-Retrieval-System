import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RAW_PAPERS_PATH = ROOT_DIR / "data" / "raw" / "papers_2015_2025.parquet"
DEFAULT_API_BASE_URL = os.getenv("PAPER_API_BASE_URL", "http://127.0.0.1:8000")
VALID_METHODS = {"bm25", "dense", "hybrid", "rerank"}
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = "hybrid"
    top_k: int = Field(10, ge=1, le=50)


app = FastAPI(title="Paper Retrieval UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def normalize_doc_id(value: Any) -> str:
    text = str(value or "").strip()
    match = ARXIV_ID_RE.search(text)
    return match.group(1) if match else text


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    return []


@lru_cache(maxsize=1)
def load_paper_metadata() -> Dict[str, Dict[str, Any]]:
    columns = ["id", "title", "abstract", "authors", "categories", "update_date"]
    df = pd.read_parquet(RAW_PAPERS_PATH, columns=columns)
    metadata: Dict[str, Dict[str, Any]] = {}

    for row in df.itertuples(index=False):
        doc_key = normalize_doc_id(row.id)
        metadata[doc_key] = {
            "id": row.id,
            "title": row.title,
            "abstract": row.abstract,
            "authors": as_list(row.authors),
            "categories": as_list(row.categories),
            "update_date": row.update_date,
            "url": row.id,
        }

    return metadata


def call_search_api(payload: SearchRequest) -> Dict[str, Any]:
    method = payload.method.lower()
    if method not in VALID_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown search method: {payload.method}")

    body = json.dumps({"query": payload.query, "top_k": payload.top_k}).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_API_BASE_URL}/search/{method}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") or str(exc)
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not reach the retrieval API. Start app/main.py on "
                f"{DEFAULT_API_BASE_URL} or set PAPER_API_BASE_URL."
            ),
        ) from exc


def enrich_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata = load_paper_metadata()
    enriched = []

    for rank, result in enumerate(results, start=1):
        doc_id = normalize_doc_id(result.get("doc_id"))
        paper = metadata.get(doc_id, {})
        title = paper.get("title") or doc_id
        url = paper.get("url") or f"https://arxiv.org/abs/{doc_id}"

        enriched.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": result.get("score"),
                "title": title,
                "url": url,
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", []),
                "categories": paper.get("categories", []),
                "update_date": paper.get("update_date"),
            }
        )

    return enriched


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/search")
def search(payload: SearchRequest) -> Dict[str, Any]:
    api_response = call_search_api(payload)
    raw_results = api_response.get("results", [])

    return {
        "status": "success",
        "query": payload.query,
        "method": payload.method.lower(),
        "count": len(raw_results),
        "results": enrich_results(raw_results),
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "api_base_url": DEFAULT_API_BASE_URL}
