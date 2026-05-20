import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Đảm bảo Python nhận diện được thư mục gốc dự án để import từ 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.inference import SearchPipeline
from src.utils.logging import get_logger
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

logger = get_logger(__name__)

# Biến toàn cục lưu trữ các thực thể mô hình sau khi nạp lên RAM
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng: Tự động nạp index/model lên RAM khi bật server."""
    logger.info("⏳ Đang nạp toàn bộ chỉ mục (Index) và Mô hình lên bộ nhớ RAM...")
    try:
        pipeline = SearchPipeline()
        # Khởi khởi tạo các bộ tìm kiếm từ mã nguồn src của bạn
        models["bm25"] = pipeline.bm25_retriever
        models["dense"] = pipeline.dense_retriever
        models["hybrid"] = pipeline.hybrid_retriever
        models["rerank"] = pipeline.reranker
        logger.info("✅ Nạp tài nguyên thành công. API đã sẵn sàng phục vụ!")
    except Exception as e:
        logger.error("❌ Lỗi nghiêm trọng khi nạp tài nguyên: %s", str(e))
        raise e
    yield
    # Giải phóng tài nguyên khi tắt server (nếu cần)
    models.clear()

app = FastAPI(
    title="arXiv Information Retrieval System",
    description="Hệ thống Tìm kiếm Ngữ nghĩa & Tái xếp hạng các bài báo khoa học arXiv (CS.AI, CS.CL, CS.IR, CS.LG)",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu trúc dữ liệu đầu vào nhận từ Client
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

# --- CÁC ENDPOINT TÌM KIẾM ---

@app.post("/search/bm25", tags=["Retrieval"])
async def search_bm25(request: SearchRequest):
    try:
        # Gọi hàm tìm kiếm thực tế từ instance của BM25Retriever
        results = models["bm25"].search(request.query, top_k=request.top_k)
        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/dense", tags=["Retrieval"])
async def search_dense(request: SearchRequest):
    try:
        results = models["dense"].search(request.query, top_k=request.top_k)
        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid", tags=["Retrieval"])
async def search_hybrid(request: SearchRequest):
    try:
        results = models["hybrid"].search(request.query, top_k=request.top_k)
        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/rerank", tags=["Reranking"])
async def search_rerank(request: SearchRequest):
    try:
        results = models["rerank"].search(request.query, top_k=request.top_k)
        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))