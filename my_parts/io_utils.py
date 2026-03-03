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


def load_metadata_list(metadata_json_path: str) -> List[Dict[str, Any]]:
    """
    Loads the saved VideoMetadata list (dicts) from *_metadata.json
    """
    p = Path(metadata_json_path)
    with p.open("r") as f:
        data = json.load(f)

    # normalize hashtags (optional)
    for row in data:
        tags = row.get("hashtags", []) or []
        row["hashtags"] = [str(t).strip().lstrip("#") for t in tags if str(t).strip() != ""]
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
