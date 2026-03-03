# my_parts/baselines.py
from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any
import random

import pandas as pd
import faiss


def naive_retrieval_from_video_index(
    user_emb,
    index,
    metadata_list,
    k: int = 10,
    pool: int = 50
) -> List[str]:
    """
    Naive Retrieval (NR) using the global video FAISS index:
    1) Retrieve top 'pool' most similar videos to the user embedding
    2) Aggregate hashtags from those videos
    3) Return top-k most common hashtags
    """
    q = user_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)

    pool = min(pool, int(index.ntotal))
    _, idxs = index.search(q, pool)

    c = Counter()
    for i in idxs[0]:
        if i < 0:
            continue
        c.update(metadata_list[i].get("hashtags", []) or [])

    return [t for t, _ in c.most_common(k)]


def baseline_global_popularity(hashtag_df: pd.DataFrame, k: int = 10) -> List[str]:
    """
    Global Popularity (GP): recommend top-k hashtags by global frequency.
    Expects columns: hashtag, frequency
    """
    return (
        hashtag_df.sort_values("frequency", ascending=False)["hashtag"]
        .head(k)
        .tolist()
    )


def baseline_niche_popularity(
    hashtag_df: pd.DataFrame,
    niche_name: str,
    k: int = 10
) -> List[str]:
    """
    Niche Popularity (NP): recommend top-k hashtags by frequency within a niche.
    Expects columns: niche_name, hashtag, frequency
    """
    sub = hashtag_df[hashtag_df["niche_name"] == niche_name]
    return (
        sub.sort_values("frequency", ascending=False)["hashtag"]
        .head(k)
        .tolist()
    )


def baseline_random_niche(
    hashtag_df: pd.DataFrame,
    niche_name: str,
    k: int = 10,
    seed: int = 0
) -> List[str]:
    """
    Random Niche (RN): randomly sample k hashtags from the niche.
    """
    random.seed(seed)
    pool = hashtag_df[hashtag_df["niche_name"] == niche_name]["hashtag"].tolist()
    if len(pool) <= k:
        return pool
    return random.sample(pool, k)


def baseline_naive_retrieval_from_search_results(
    search_results: List[Dict[str, Any]],
    k: int = 10
) -> List[str]:
    """
    Alternate NR if you already retrieved videos via a search API:
    given top retrieved videos, return most common hashtags among them.
    """
    c = Counter()
    for r in search_results:
        c.update(r.get("hashtags", []) or [])
    return [t for t, _ in c.most_common(k)]
