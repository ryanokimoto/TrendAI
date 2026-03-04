# my_parts/ranker.py
from __future__ import annotations

from typing import Dict, List, Tuple


def rank_hashtags(
    candidates: List[Dict],  # from NicheRetrievalEngine.retrieve -> {"hashtag","score"}
    niche_name: str,
    vel_norm: Dict[Tuple[str, str], float],         # (niche, hashtag)->[0,1]
    freq_norm: Dict[str, float],                    # hashtag->[0,1]
    w_sim: float = 0.6,
    w_vel: float = 0.3,
    w_sat: float = 0.1,
    top_k: int = 10
):
    scored = []
    for c in candidates:
        tag = c["hashtag"]
        sim = float(c["score"])
        vel = vel_norm.get((niche_name, tag), 0.0)
        sat = freq_norm.get(tag, 0.0)  # higher = more saturated
        final = w_sim * sim + w_vel * vel - w_sat * sat
        scored.append({
            "hashtag": tag,
            "final_score": final,
            "sim": sim,
            "vel": vel,
            "sat": sat,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]
