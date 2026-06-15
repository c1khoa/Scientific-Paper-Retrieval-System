import json
import pandas as pd
import random
import time
import feedparser
from pathlib import Path
from urllib.parse import quote
from tqdm import tqdm

RAW_DIR = "../data/raw"
DATASET_PARQUET = "../data/raw/papers_2015_2025.parquet"
QUERIES_PATH = "../data/ground_truth/queries.parquet"
QRELS_PATH = "../data/ground_truth/qrels.parquet"

records = []
seen_ids = set()

files = list(Path(RAW_DIR).rglob("*.jsonl"))

print("Total files:", len(files))

for file in tqdm(files):

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)

            paper_id = paper.get("id")

            if paper_id in seen_ids:
                continue

            seen_ids.add(paper_id)

            records.append({
                "id": paper_id,
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": paper.get("authors"),
                "categories": paper.get("categories"),
                "update_date": paper.get("update_date")
            })

print("Unique papers:", len(records))

df = pd.DataFrame(records)
df.to_parquet(DATASET_PARQUET, index=False)

print("Saved:", DATASET_PARQUET)

topics = [
    "neural machine translation", "information retrieval", "document ranking",
    "transformer language models", "contrastive learning", "question answering",
    "document retrieval", "large language models", "multilingual NLP",
    "neural search", "deep learning optimization", "scientific paper recommendation",
    "semantic similarity", "knowledge graph reasoning", "dense passage retrieval",
    "self supervised learning", "text classification", "topic modeling",
    "document clustering", "neural reranking", "representation learning",
    "language understanding", "cross lingual retrieval", "neural embeddings"
]

tasks = [
    "models", "methods", "approaches", 
    "systems", "techniques", "algorithms"
]

contexts = [
    "for academic search", "for scientific papers", 
    "for research literature", "in large scale datasets",
    "for multilingual corpora", "in neural IR systems", 
    "for NLP benchmarks", "in academic recommender systems"
]

length_distribution = [
    (1, 35),
    (2, 60),
    (3, 90),
    (4, 60),
    (5, 35),
    (6, 20),
]

vocab = list(set(" ".join(topics).split()))

def build_query(length):

    if length == 1:
        return random.choice(vocab)

    topic = random.choice(topics)
    words = topic.split()

    random.shuffle(words)

    words = words[:min(len(words), length)]

    if len(words) < length and random.random() < 0.7:
        words.append(random.choice(tasks))

    if len(words) < length and random.random() < 0.5:
        words += random.choice(contexts).split()

    return " ".join(words[:length])


def token_overlap(q1, q2):

    s1 = set(q1.split())
    s2 = set(q2.split())

    return len(s1 & s2) / len(s1 | s2)


queries = []
texts = []
qid = 1

max_attempts = 40

for length, count in length_distribution:

    for _ in tqdm(range(count), desc=f"length={length}"):

        attempts = 0

        while True:

            q = build_query(length)
            attempts += 1

            duplicate = False

            if length >= 3:
                for existing in texts:
                    if token_overlap(q, existing) >= 0.5:
                        duplicate = True
                        break

            if not duplicate or attempts > max_attempts:
                break

        texts.append(q)

        queries.append({
            "query_id": f"q{qid}",
            "text": q
        })

        qid += 1

queries_df = pd.DataFrame(queries)

queries_df.to_parquet(QUERIES_PATH, index=False)

print("Saved:", QUERIES_PATH)
print("Total queries:", len(queries_df))

ARXIV_API = "http://export.arxiv.org/api/query"

ALLOWED = {"cs.AI","cs.CL","cs.IR","cs.LG"}

START_YEAR = 2015
END_YEAR = 2025

TOP_K = 50

papers = pd.read_parquet(DATASET_PARQUET)

dataset_ids = set(
    papers["id"].str.split("/abs/").str[-1].str.split("v").str[0]
)

print("Corpus size:", len(dataset_ids))


def arxiv_search(query, start):

    query_encoded = quote(query)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=abs:{query_encoded}"
        f"&start={start}"
        f"&max_results=100"
        f"&sortBy=relevance"
    )

    feed = feedparser.parse(url)

    return feed.entries


print("Queries:", len(queries_df))

qrels = []

for _, q in tqdm(queries_df.iterrows(), total=len(queries_df)):

    qid = q["query_id"]
    text = q["text"]

    start = 0
    collected = []

    while len(collected) < TOP_K:

        entries = arxiv_search(text, start)

        if len(entries) == 0:
            break

        for e in entries:

            arxiv_id = e.id.split("/abs/")[-1].split("v")[0]

            year = int(e.published[:4])

            cats = {t["term"] for t in e.tags}

            if year < START_YEAR or year > END_YEAR:
                continue

            if not (cats & ALLOWED):
                continue

            if arxiv_id not in dataset_ids:
                continue

            collected.append(arxiv_id)

            if len(collected) >= TOP_K:
                break

        start += 100

        time.sleep(0.5)

    for i, doc in enumerate(collected):

        if i < 6:
            rel = 3
        elif i < 21:
            rel = 2
        else:
            rel = 1

        qrels.append({
            "query_id": qid,
            "doc_id": doc,
            "relevance": rel
        })


qrels_df = pd.DataFrame(qrels)

qrels_df.to_parquet(QRELS_PATH, index=False)

print("Saved qrels.parquet")