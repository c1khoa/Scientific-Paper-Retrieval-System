from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import DOC_IDS_PATH, FAISS_INDEX_PATH, PREPROCESSED_DATA_PATH
from src.config.settings import BATCH_SIZE, E5_CACHE_DIR, E5_MODEL_NAME
from src.embedding.e5_encoder import E5Encoder
from src.indexing.faiss_manager import FaissManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_dense_index(
    input_path=PREPROCESSED_DATA_PATH,
    index_path=FAISS_INDEX_PATH,
    doc_ids_path=DOC_IDS_PATH,
) -> FaissManager:
    logger.info("Starting dense index build")
    logger.info("Reading preprocessed data from %s", input_path)
    df = pd.read_parquet(input_path)
    texts = df["text"].fillna("").astype(str).tolist()
    doc_ids = df["doc_id"].to_numpy()
    logger.info("Loaded %s documents", len(texts))

    encoder = E5Encoder(E5_MODEL_NAME, cache_dir=E5_CACHE_DIR)
    embeddings = encoder.encode_documents(texts, batch_size=BATCH_SIZE)

    faiss_manager = FaissManager()
    faiss_manager.build(embeddings)
    faiss_manager.save(index_path)

    doc_ids_path = Path(doc_ids_path)
    doc_ids_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving document ids to %s", doc_ids_path)
    np.save(doc_ids_path, doc_ids)

    logger.info("Dense index build finished")
    return faiss_manager


def main() -> None:
    build_dense_index()


if __name__ == "__main__":
    main()
