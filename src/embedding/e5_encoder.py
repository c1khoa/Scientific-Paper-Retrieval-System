import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class E5Encoder:
    """Encode documents and queries using the E5 prompt format."""

    DOCUMENT_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(
        self,
        model_name: str = "intfloat/e5-base",
        device: str | None = None,
        cache_dir: str = "models/e5-base",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading E5 model '%s' on device '%s'", model_name, self.device)
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_dir,
        )
        logger.info("E5 model loaded")

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> np.ndarray:
        prefixed_texts = [self.DOCUMENT_PREFIX + str(text) for text in texts]
        logger.info(
            "Encoding %s documents with batch_size=%s",
            len(prefixed_texts),
            batch_size,
        )
        embeddings = [
            self.model.encode(
                prefixed_texts[start : start + batch_size],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for start in tqdm(
                range(0, len(prefixed_texts), batch_size),
                desc="Encoding docs",
            )
        ]

        if not embeddings:
            logger.info("No documents to encode")
            return np.empty((0, 0), dtype=np.float32)

        stacked = np.vstack(embeddings).astype(np.float32)
        logger.info("Document encoding complete: shape=%s", stacked.shape)
        return stacked

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefixed_queries = [self.QUERY_PREFIX + str(query) for query in queries]
        logger.info("Encoding %s queries", len(prefixed_queries))
        embeddings = self.model.encode(
            prefixed_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = embeddings.astype(np.float32)
        logger.info("Query encoding complete: shape=%s", embeddings.shape)
        return embeddings
