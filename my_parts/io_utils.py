# my_parts/io_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import faiss


def load_hashtag_to_niche(csv_path: str = "hashtag_to_niche.csv") -> pd.DataFrame:
    """
    Expected columns: hashtag, cluster, niche_name, frequency
    """
    df = pd.read_csv(csv_path)
    # normalize hashtags (optional but helpful)
    df["hashtag"] = df["hashtag"].astype(str).str.strip().str.lstrip("#")
    return df


# my_parts/io_utils.py (replace ONLY load_metadata_list with this)
def load_metadata_list(
    metadata_json_path: str,
    hashtags_csv_path: str | None = None,
    id_col: str = "id",
    challenges_col: str = "challenges",
) -> List[Dict[str, Any]]:
    """
    Loads the saved VideoMetadata list (dicts) from *_metadata.json.
    Optionally injects hashtags from a CSV (e.g., batch1.csv) that contains a
    `challenges` column like ["puppy","pets"] or a list of dicts with {"title":...}.

    This avoids editing create_embeddings.py.
    """
    p = Path(metadata_json_path)
    with p.open("r") as f:
        data = json.load(f)

    # normalize hashtags field (keep but won't help if it's empty)
    for row in data:
        tags = row.get("hashtags", []) or []
        if isinstance(tags, list):
            row["hashtags"] = [str(t).strip().lstrip("#") for t in tags if str(t).strip()]
        else:
            row["hashtags"] = []

    if hashtags_csv_path is None:
        return data

    # --- inject hashtags from CSV ---
    df = pd.read_csv(hashtags_csv_path)

    import ast

    def parse_challenges(x) -> List[str]:
        if pd.isna(x):
            return []
        s = str(x).strip()
        try:
            v = ast.literal_eval(s)
        except Exception:
            return []

        out = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, dict):
                    t = item.get("title", "")
                    if isinstance(t, str) and t.strip():
                        out.append(t.strip())
        return [h.strip().lstrip("#") for h in out if str(h).strip()]

    # build mapping from video_id -> hashtags from the CSV
    vid2tags: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        vid = str(r.get(id_col, "")).strip()
        if not vid:
            continue
        tags = parse_challenges(r.get(challenges_col))
        if tags:
            vid2tags[vid] = tags

    # apply into metadata list
    updated = 0
    for row in data:
        vid = str(row.get("video_id", "")).strip()
        if vid in vid2tags:
            row["hashtags"] = vid2tags[vid]
            updated += 1

    print(f"[io_utils] Injected hashtags for {updated} / {len(data)} videos from {hashtags_csv_path}")
    return data


def load_faiss_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)


def reconstruct_embeddings(index: faiss.Index) -> np.ndarray:
    """
    Reconstruct all vectors from a FAISS flat index (IndexFlatIP / IndexFlatL2).
    """
    n = index.ntotal
    d = index.d

    embs = np.zeros((n, d), dtype=np.float32)

    for i in range(n):
        vec = np.zeros(d, dtype=np.float32)
        index.reconstruct(i, vec)
        embs[i] = vec

    return embs


def build_video_id_to_meta(metadata_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {m["video_id"]: m for m in metadata_list if "video_id" in m}


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
