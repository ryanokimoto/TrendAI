# my_parts/run_pipeline.py
from __future__ import annotations

import numpy as np

from tiktok_search_system import TikTokEmbeddingSearch
from tiktok_retrieval_system import NicheRetrievalEngine  # note: ensure filename matches import
from my_parts.io_utils import (
    load_hashtag_to_niche,
    load_metadata_list,
    load_faiss_index,
    reconstruct_embeddings,
)
from my_parts.baselines import (
    baseline_global_popularity,
    baseline_niche_popularity,
    baseline_random_niche,
    baseline_naive_retrieval_from_search_results,
)
from my_parts.trend_metrics import compute_trend_velocity, normalize_scores
from my_parts.ranker import rank_hashtags
from my_parts.explain import explain_recommendation


def main():
    # ---- paths: match your repo ----
    hashtag_csv = "hashtag_to_niche.csv"
    index_path = "tiktok_data/tiktok_vision_index.index"
    metadata_path = "tiktok_data/tiktok_vision_index_metadata.json"

    hashtag_df = load_hashtag_to_niche(hashtag_csv)
    metadata_list = load_metadata_list(metadata_path)

    # maps for trends + saturation
    hashtag_to_niche = dict(zip(hashtag_df["hashtag"], hashtag_df["niche_name"]))
    freq = dict(zip(hashtag_df["hashtag"], hashtag_df["frequency"]))
    freq_norm = normalize_scores(freq)

    # ---- compute trend velocity (Part 6) ----
    vel = compute_trend_velocity(metadata_list, hashtag_to_niche, window_days=7)
    vel_norm = normalize_scores(vel)

    # ---- reconstruct video embeddings from FAISS index ----
    index = load_faiss_index(index_path)
    video_embeddings = reconstruct_embeddings(index)  # (N, 384)

    # ---- build niche retrieval engine (uses teammate code) ----
    engine = NicheRetrievalEngine(embedding_dim=video_embeddings.shape[1])
    engine.compute_hashtag_representative_vectors(video_embeddings, metadata_list)
    engine.build_niche_faiss_indices(hashtag_df)

    # ---- user input ----
    user_text = input("\nDescribe your video / query: ").strip()

    # Use their semantic search to get top videos (also used for naive baseline)
    search_sys = TikTokEmbeddingSearch(model_name="all-MiniLM-L6-v2", data_dir="./tiktok_data")
    # We can load the SAME metadata/index if you saved via TikTokEmbeddingSearch.save(name=...)
    # But your repo artifacts are named tiktok_vision_index.*, so we use engine for hashtag retrieval.
    # Still, we can embed user text with the SentenceTransformer model inside search_sys:
    user_emb = search_sys.model.encode([user_text], convert_to_numpy=True).astype("float32")[0]

    # ---- infer niche using quick baseline retrieval from all niches ----
    # simplest: pick a niche by naive retrieval from search results if you have a saved TikTokEmbeddingSearch index.
    # If you *don't* have a TikTokEmbeddingSearch saved under a name, we can just default niche.
    # Here: choose niche by taking the top niche among candidate hashtags from GLOBAL popularity inside metadata.
    # Practical approach: ask user niche name or pick a common niche.
    niche_name = input("Niche name (must match hashtag_to_niche.csv niche_name): ").strip()

    # ---- retrieval from niche index ----
    candidates = engine.retrieve(user_emb, niche_name, top_k=80)

    # ---- rank (Part 7) ----
    ranked = rank_hashtags(
        candidates=candidates,
        niche_name=niche_name,
        vel_norm=vel_norm,
        freq_norm=freq_norm,
        w_sim=0.6, w_vel=0.3, w_sat=0.1,
        top_k=10
    )

    # ---- explain (Part 8) ----
    print("\n=== Final Recommendations (Ranked) ===")
    for r in ranked:
        print(explain_recommendation(
            tag=r["hashtag"],
            niche_name=niche_name,
            sim=r["sim"],
            vel=r["vel"],
            sat=r["sat"],
        ))

    # ---- baselines (1,2,4 + RN) ----
    print("\n=== Baselines ===")
    print("GP:", baseline_global_popularity(hashtag_df, k=10))
    print("NP:", baseline_niche_popularity(hashtag_df, niche_name=niche_name, k=10))
    print("RN:", baseline_random_niche(hashtag_df, niche_name=niche_name, k=10, seed=0))

    # Naive retrieval baseline from your niche candidates:
    # (closest to your NR definition while staying scalable)
    print("NR:", [c["hashtag"] for c in candidates[:10]])


if __name__ == "__main__":
    main()
