from pathlib import Path

import faiss
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaissManager:
    def __init__(
        self,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.index = None
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        embeddings = embeddings.astype("float32")
        dimension = embeddings.shape[1]
        logger.info(
            "Building FAISS HNSW index: documents=%s dimension=%s hnsw_m=%s",
            embeddings.shape[0],
            dimension,
            self.hnsw_m,
        )
        index = faiss.IndexHNSWFlat(
            dimension,
            self.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(embeddings)

        self.index = index
        logger.info("FAISS index built")

    def search(self, query_embeddings: np.ndarray, top_k: int):
        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded.")

        logger.info(
            "Searching FAISS index: queries=%s top_k=%s",
            query_embeddings.shape[0],
            top_k,
        )
        return self.index.search(query_embeddings.astype("float32"), top_k)

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty FAISS index.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving FAISS index to %s", output_path)
        faiss.write_index(self.index, str(output_path))
        logger.info("FAISS index saved")

    def load(self, path: str | Path) -> None:
        input_path = Path(path)
        logger.info("Loading FAISS index from %s", input_path)
        self.index = faiss.read_index(str(input_path))
        logger.info("FAISS index loaded")
