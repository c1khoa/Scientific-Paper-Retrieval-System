from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ROOT_DIR / "results"

DENSE_DIR = ARTIFACTS_DIR / "dense"
SPARSE_DIR = ARTIFACTS_DIR / "sparse"

FAISS_INDEX_PATH = DENSE_DIR / "paper_index.faiss"
DOC_IDS_PATH = DENSE_DIR / "doc_ids.npy"

BM25_PATH = SPARSE_DIR / "bm25.pkl"
TOKENIZED_DOCS_PATH = SPARSE_DIR / "tokenized_docs.pkl"

PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed" / "papers_preprocessed.parquet"

RUN_DENSE = RESULTS_DIR / "run_dense.parquet"
RUN_BM25 = RESULTS_DIR / "run_bm25.parquet"
RUN_HYBRID = RESULTS_DIR / "run_hybrid.parquet"
RUN_RERANK = RESULTS_DIR / "run_rerank.parquet"

FULL_EVAL = RESULTS_DIR / "full_evaluation_report.csv"


def ensure_artifact_dirs() -> None:
    """Create index artifact directories if they do not exist."""
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
