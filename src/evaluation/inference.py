import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

sys.path.append("..")

from src.config.paths import BM25_PATH, DOC_IDS_PATH, FAISS_INDEX_PATH, RUN_DENSE, RUN_BM25, RUN_HYBRID, RUN_RERANK
from src.config.settings import E5_CACHE_DIR, E5_MODEL_NAME, TOP_K, CANDIDATE_K
from src.embedding.e5_encoder import E5Encoder
from src.indexing.bm25_index import BM25Index
from src.indexing.faiss_manager import FaissManager
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.bm25_e5_reranker import BM25E5Reranker
from src.retrievers.e5_retriever import E5Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)
QUERIES_PARQUET_PATH = "data/ground_truth/queries.parquet" 


class SearchPipeline:
    def __init__(self):
        self.doc_ids = np.load(DOC_IDS_PATH, allow_pickle=True)
        self.encoder = E5Encoder(E5_MODEL_NAME, cache_dir=E5_CACHE_DIR)
        
        self.faiss_manager = FaissManager()
        self.faiss_manager.load(FAISS_INDEX_PATH)
        
        self.bm25_index = BM25Index()
        self.bm25_index.load(BM25_PATH)
        
        self.dense_retriever = E5Retriever(self.encoder, self.faiss_manager, self.doc_ids)
        self.bm25_retriever = BM25Retriever(self.bm25_index, self.doc_ids)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.bm25_retriever)
        self.reranker = BM25E5Reranker(
            self.bm25_index,
            self.encoder,
            self.faiss_manager,
            self.doc_ids,
        )


def process_search_results(query_id, search_results, top_k):
    rows = []
    for res in search_results:
        # Tự động parse dựa theo kiểu dữ liệu thực tế của từng retriever
        if isinstance(res, dict):
            doc_id = res["doc_id"]
            score = float(res["score"])
        elif isinstance(res, (tuple, list)) and len(res) >= 2:
            doc_id = res[0]
            score = float(res[1])
        else:
            doc_id = res
            score = float(top_k - len(rows))
            
        rows.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "score": score
        })
    return rows


def run_batch_inference(pipeline: SearchPipeline, df_queries: pd.DataFrame, top_k: int, candidate_k: int):
    dense_rows, bm25_rows, hybrid_rows, rerank_rows = [], [], [], []
    
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc="Inference Progress"):
        q_id = row["query_id"]
        q_text = row["text"]
        
        dense_res = pipeline.dense_retriever.search(q_text, top_k=top_k)
        dense_rows.extend(process_search_results(q_id, dense_res, top_k))
        
        bm25_res = pipeline.bm25_retriever.search(q_text, top_k=top_k)
        bm25_rows.extend(process_search_results(q_id, bm25_res, top_k))
        
        hybrid_res = pipeline.hybrid_retriever.search(q_text, top_k=top_k)
        hybrid_rows.extend(process_search_results(q_id, hybrid_res, top_k))
        
        rerank_res = pipeline.reranker.search(q_text, top_k=top_k, candidate_k=candidate_k)
        rerank_rows.extend(process_search_results(q_id, rerank_res, top_k))
        
    return dense_rows, bm25_rows, hybrid_rows, rerank_rows


def post_process_and_sort(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["query_id", "doc_id", "score"])
    
    df["doc_id"] = df["doc_id"].astype(str)
    df["query_id"] = df["query_id"].astype(str)
    
    return df.sort_values(by=["query_id", "score"], ascending=[True, False]).reset_index(drop=True)


def debug_single_query(pipeline: SearchPipeline, query_id="q1", text="test query"):
    dense_res = pipeline.dense_retriever.search(text, top_k=2)
    hybrid_res = pipeline.hybrid_retriever.search(text, top_k=2)
    print(f"Cấu trúc Dense mẫu:  {dense_res}")
    print(f"Cấu trúc Hybrid mẫu: {hybrid_res}")


def main():
    if not os.path.exists(QUERIES_PARQUET_PATH):
        return

    df_queries = pd.read_parquet(QUERIES_PARQUET_PATH)
    pipeline = SearchPipeline()
    
    if not df_queries.empty:
        first_row = df_queries.iloc[0]
        debug_single_query(pipeline, query_id=first_row["query_id"], text=first_row["text"])
    
    dense_r, bm25_r, hybrid_r, rerank_r = run_batch_inference(
        pipeline, df_queries, top_k=TOP_K, candidate_k=CANDIDATE_K
    )
    
    df_dense  = post_process_and_sort(dense_r)
    df_bm25   = post_process_and_sort(bm25_r)
    df_hybrid = post_process_and_sort(hybrid_r)
    df_rerank = post_process_and_sort(rerank_r)
    
    df_dense.to_parquet(RUN_DENSE, index=False)
    df_bm25.to_parquet(RUN_BM25, index=False)
    df_hybrid.to_parquet(RUN_HYBRID, index=False)
    df_rerank.to_parquet(RUN_RERANK, index=False)


if __name__ == "__main__":
    main()