import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class E5Retriever:
    def __init__(self, encoder, faiss_index, doc_ids) -> None:
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.doc_ids = np.asarray(doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        logger.info("Running dense retrieval: top_k=%s query=%r", top_k, query)
        query_embedding = self.encoder.encode_queries([query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = [
            {
                "doc_id": self.doc_ids[idx],
                "score": float(score),
            }
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
        logger.info("Dense retrieval complete: results=%s", len(results))
        return results
