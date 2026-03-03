# my_parts/baselines.py
from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any
import random
import pandas as pd


def baseline_global_popularity(hashtag_df: pd.DataFrame, k: int = 10) -> List[str]:
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
    Naive Retrieval (NR):
      given top retrieved videos, return most common hashtags among them.
    """
    c = Counter()
    for r in search_results:
        c.update(r.get("hashtags", []) or [])
    return [t for t, _ in c.most_common(k)]
