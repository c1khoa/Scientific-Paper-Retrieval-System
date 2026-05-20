import argparse

import numpy as np

from src.config.paths import BM25_PATH, DOC_IDS_PATH, FAISS_INDEX_PATH
from src.config.settings import E5_CACHE_DIR, E5_MODEL_NAME, TOP_K
from src.embedding.e5_encoder import E5Encoder
from src.indexing.bm25_index import BM25Index
from src.indexing.faiss_manager import FaissManager
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.bm25_e5_reranker import BM25E5Reranker
from src.retrievers.e5_retriever import E5Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_doc_ids():
    logger.info("Loading document ids from %s", DOC_IDS_PATH)
    doc_ids = np.load(DOC_IDS_PATH, allow_pickle=True)
    logger.info("Loaded %s document ids", len(doc_ids))
    return doc_ids


def load_dense_retriever(doc_ids) -> E5Retriever:
    encoder = E5Encoder(E5_MODEL_NAME, cache_dir=E5_CACHE_DIR)

    faiss_manager = FaissManager()
    faiss_manager.load(FAISS_INDEX_PATH)

    return E5Retriever(encoder, faiss_manager, doc_ids)


def load_bm25_retriever(doc_ids) -> BM25Retriever:
    bm25_index = BM25Index()
    bm25_index.load(BM25_PATH)

    return BM25Retriever(bm25_index, doc_ids)


def load_hybrid_retriever(dense_retriever, bm25_retriever) -> HybridRetriever:
    return HybridRetriever(dense_retriever, bm25_retriever)


def load_bm25_e5_reranker(
    doc_ids,
    dense_retriever=None,
    bm25_retriever=None,
) -> BM25E5Reranker:
    if dense_retriever is None:
        dense_retriever = load_dense_retriever(doc_ids)

    if bm25_retriever is None:
        bm25_retriever = load_bm25_retriever(doc_ids)

    return BM25E5Reranker(
        bm25_retriever.bm25_index,
        dense_retriever.encoder,
        dense_retriever.faiss_index,
        doc_ids,
    )


doc_ids = load_doc_ids()
dense_retriever = None
bm25_retriever = None

def inference_dense(doc_ids, query, top_k=50):
    dense_retriever = load_dense_retriever(doc_ids)
    dense_results = dense_retriever.search(query, top_k=top_k)
    return dense_results

def inference_bm25(doc_ids, query, top_k=50):
    bm25_retriever = load_bm25_retriever(doc_ids)
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    return bm25_results

def inference_hybrid(doc_ids, query, top_k=50):
    dense_retriever = dense_retriever or load_dense_retriever(doc_ids)
    bm25_retriever = bm25_retriever or load_bm25_retriever(doc_ids)
    hybrid_retriever = load_hybrid_retriever(dense_retriever, bm25_retriever)
    hybrid_results = hybrid_retriever.search(query, top_k=top_k)
    return hybrid_results
    
def inference_rerank(doc_ids, query, top_k=50, candidate_k=100):
    reranker = load_bm25_e5_reranker(
        doc_ids,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )
    reranked_results = reranker.search(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    return reranked_results