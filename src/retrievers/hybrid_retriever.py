from src.config.settings import RRF_K
from src.utils.rrf import reciprocal_rank_fusion
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, rrf_k: int = RRF_K) -> None:
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        logger.info("Running hybrid retrieval: top_k=%s query=%r", top_k, query)

        dense_results = self.dense.search(query, top_k)
        sparse_results = self.sparse.search(query, top_k)

        logger.info(
            "Fusing rankings: dense_results=%s sparse_results=%s",
            len(dense_results),
            len(sparse_results),
        )

        fusion_scores = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

        results = sorted(
            fusion_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        formatted_results = [
            {
                "doc_id": doc_id,
                "score": float(score),
            }
            for doc_id, score in results
        ]

        logger.info("Hybrid retrieval complete: results=%s", len(formatted_results))

        return formatted_results
