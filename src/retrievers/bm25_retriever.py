import numpy as np

from src.embedding.tokenization import tokenize
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self, bm25_index, doc_ids) -> None:
        self.bm25_index = bm25_index
        self.doc_ids = np.asarray(doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        logger.info("Running BM25 retrieval: top_k=%s query=%r", top_k, query)
        tokenized_query = tokenize(query)
        scores = self.bm25_index.search(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            {
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]
        logger.info("BM25 retrieval complete: results=%s", len(results))
        return results
