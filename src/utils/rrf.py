def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> dict:
    scores = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc["doc_id"]
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += 1 / (k + rank + 1)

    return scores
