import numpy as np

from src.embedding.tokenization import tokenize
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25E5Reranker:
    """Retrieve candidates with BM25, then rerank them with E5 similarity."""

    def __init__(self, bm25_index, encoder, faiss_index, doc_ids) -> None:
        self.bm25_index = bm25_index
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.doc_ids = np.asarray(doc_ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
    ) -> list[dict]:
        logger.info(
            "Running BM25->E5 rerank: candidate_k=%s top_k=%s query=%r",
            candidate_k,
            top_k,
            query,
        )

        tokenized_query = tokenize(query)
        bm25_scores = self.bm25_index.search(tokenized_query)
        candidate_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

        if len(candidate_indices) == 0:
            logger.info("BM25 returned no candidates")
            return []

        query_embedding = self.encoder.encode_queries([query])[0]
        candidate_embeddings = np.vstack(
            [
                self.faiss_index.index.reconstruct(int(index))
                for index in candidate_indices
            ]
        ).astype(np.float32)

        e5_scores = candidate_embeddings @ query_embedding
        ranked_positions = np.argsort(e5_scores)[::-1][:top_k]

        results = [
            {
                "doc_id": self.doc_ids[candidate_indices[position]],
                "score": float(e5_scores[position]),
                "bm25_score": float(bm25_scores[candidate_indices[position]]),
            }
            for position in ranked_positions
        ]
        logger.info("BM25->E5 rerank complete: results=%s", len(results))
        return results
