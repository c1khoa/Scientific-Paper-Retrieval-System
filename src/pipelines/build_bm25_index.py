import pickle
from pathlib import Path

import pandas as pd

from src.config.paths import BM25_PATH, PREPROCESSED_DATA_PATH, TOKENIZED_DOCS_PATH
from src.indexing.bm25_index import BM25Index
from src.embedding.tokenization import tokenize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_bm25_index(
    input_path=PREPROCESSED_DATA_PATH,
    index_path=BM25_PATH,
    tokenized_docs_path=TOKENIZED_DOCS_PATH,
) -> BM25Index:
    logger.info("Starting BM25 build")
    logger.info("Reading preprocessed data from %s", input_path)
    df = pd.read_parquet(input_path)
    texts = df["text"].fillna("").astype(str).tolist()
    logger.info("Loaded %s documents", len(texts))

    logger.info("Tokenizing documents")
    tokenized_docs = [tokenize(text) for text in texts]
    logger.info("Tokenization complete")

    bm25 = BM25Index()
    bm25.build(tokenized_docs)
    bm25.save(index_path)

    tokenized_docs_path = Path(tokenized_docs_path)
    tokenized_docs_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tokenized documents to %s", tokenized_docs_path)
    with tokenized_docs_path.open("wb") as file:
        pickle.dump(tokenized_docs, file)

    logger.info("BM25 build finished")
    return bm25


def main() -> None:
    build_bm25_index()


if __name__ == "__main__":
    main()
