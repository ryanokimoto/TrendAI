# run_pipeline_ui.py
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import numpy as np

from tiktok_search_system import TikTokEmbeddingSearch
from tiktok_retrieval_system import NicheRetrievalEngine

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
)

from my_parts.trend_metrics import compute_trend_velocity, normalize_scores
from my_parts.ranker import rank_hashtags
from my_parts.explain import explain_recommendation


# --------- LOAD DATA ONCE ---------

print("Loading data...")

hashtag_csv = "hashtag_to_niche.csv"
index_path = "tiktok_data/tiktok_vision_index.index"
metadata_path = "tiktok_data/tiktok_vision_index_metadata.json"

hashtag_df = load_hashtag_to_niche(hashtag_csv)
metadata_list = load_metadata_list(metadata_path, hashtags_csv_path="batch1.csv")

hashtag_to_niche = dict(zip(hashtag_df["hashtag"], hashtag_df["niche_name"]))
freq = dict(zip(hashtag_df["hashtag"], hashtag_df["frequency"]))
freq_norm = normalize_scores(freq)

vel = compute_trend_velocity(metadata_list, hashtag_to_niche, window_days=7)
vel_norm = normalize_scores(vel)

index = load_faiss_index(index_path)
video_embeddings = reconstruct_embeddings(index)

engine = NicheRetrievalEngine(embedding_dim=video_embeddings.shape[1])
engine.compute_hashtag_representative_vectors(video_embeddings, metadata_list)
engine.build_niche_faiss_indices(hashtag_df)

search_sys = TikTokEmbeddingSearch(
    model_name="all-MiniLM-L6-v2",
    data_dir="./tiktok_data"
)

print("System ready.")


# --------- MAIN FUNCTION ---------

def recommend_hashtags(user_text, niche_name):

    user_emb = search_sys.model.encode(
        [user_text],
        convert_to_numpy=True
    ).astype("float32")[0]

    candidates = engine.retrieve(user_emb, niche_name, top_k=80)

    ranked = rank_hashtags(
        candidates=candidates,
        niche_name=niche_name,
        vel_norm=vel_norm,
        freq_norm=freq_norm,
        w_sim=0.6,
        w_vel=0.3,
        w_sat=0.1,
        top_k=10
    )

    explanations = []

    for r in ranked:
        explanations.append(
            explain_recommendation(
                tag=r["hashtag"],
                niche_name=niche_name,
                sim=r["sim"],
                vel=r["vel"],
                sat=r["sat"],
            )
        )

    gp = baseline_global_popularity(hashtag_df, k=10)
    npb = baseline_niche_popularity(hashtag_df, niche_name=niche_name, k=10)
    rn = baseline_random_niche(hashtag_df, niche_name=niche_name, k=10, seed=0)

    result = "### 🔥 Recommended Hashtags\n\n"
    result += "\n".join(explanations)

    result += "\n\n---\n"
    result += f"\n**Global Popularity:** {gp}"
    result += f"\n\n**Niche Popularity:** {npb}"
    result += f"\n\n**Random Niche:** {rn}"

    return result


# --------- GRADIO UI ---------

demo = gr.Interface(
    fn=recommend_hashtags,
    inputs=[
        gr.Textbox(label="Describe your TikTok video"),
        gr.Textbox(label="Niche name")
    ],
    outputs=gr.Markdown(),
    title="TikTok Hashtag Recommendation System",
    description="Enter a description of your TikTok video and a niche."
)

demo.launch(share=True)
