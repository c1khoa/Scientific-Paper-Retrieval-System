import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Index:
    def __init__(self) -> None:
        self.index: BM25Okapi | None = None

    def build(self, tokenized_docs: list[list[str]]) -> None:
        logger.info("Building BM25 index for %s documents", len(tokenized_docs))
        self.index = BM25Okapi(tokenized_docs)
        logger.info("BM25 index built")

    def search(self, tokenized_query: list[str]):
        if self.index is None:
            raise RuntimeError("BM25 index has not been built or loaded.")

        logger.info("Searching BM25 index with %s query tokens", len(tokenized_query))
        return self.index.get_scores(tokenized_query)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving BM25 index to %s", output_path)
        with output_path.open("wb") as file:
            pickle.dump(self.index, file)
        logger.info("BM25 index saved")

    def load(self, path: str | Path) -> None:
        input_path = Path(path)
        logger.info("Loading BM25 index from %s", input_path)
        with input_path.open("rb") as file:
            self.index = pickle.load(file)
        logger.info("BM25 index loaded")
