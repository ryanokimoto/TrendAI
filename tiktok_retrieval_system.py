"""
For similarity search, mean pooling is implemented to aggregate video embeddings into representative hashtag embeddings.
tiktok_search_system.py generates the video-level embeddings, and this will consume them and computes a single vector per hashtag.
For retrieval, there is a multi-index architecture where each niche has its own dedicated FAISS index instead of using a global search space.
Given a user embedding and a predicted niche ID, the system selects the corresponding niche index and returns the top relevant hashtags based on cosine similarity.
A mock test pipeline is also implemented, so the module can run independently for demonstration.
To fully integrate into a main pipeline, this module only requires video_embeddings, video_metadata, and hashtag_to_niche.csv.
"""

import numpy as np
import pandas as pd
import faiss
from collections import defaultdict

class NicheRetrievalEngine:

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.niche_indices = {}
        self.niche_hashtag_maps = {}
        self.hashtag_vectors = {}

    def compute_hashtag_representative_vectors(self, video_embeddings, video_metadata_list):
        tag_to_vectors = defaultdict(list)

        for i, meta in enumerate(video_metadata_list):
            v_tags = meta['hashtags']
            v_emb = video_embeddings[i]

            for tag in v_tags:
                tag_to_vectors[tag].append(v_emb)

        for tag, vecs in tag_to_vectors.items():
            self.hashtag_vectors[tag] = np.mean(vecs, axis=0)

        print(f"Computed {len(self.hashtag_vectors)} hashtag vectors")

    def build_niche_faiss_indices(self, hashtag_to_niche_df):
        niche_groups = hashtag_to_niche_df.groupby('niche_name')

        for niche_name, group in niche_groups:
            tags_in_niche = group['hashtag'].tolist()
            vectors = []
            valid_tags = []

            for tag in tags_in_niche:
                if tag in self.hashtag_vectors:
                    vectors.append(self.hashtag_vectors[tag])
                    valid_tags.append(tag)

            if not vectors:
                continue

            matrix = np.vstack(vectors).astype('float32')
            faiss.normalize_L2(matrix)

            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(matrix)

            self.niche_indices[niche_name] = index
            self.niche_hashtag_maps[niche_name] = valid_tags

        print(f"Built {len(self.niche_indices)} niche indices")

    def retrieve(self, user_video_emb, niche_name, top_k=50):
        if niche_name not in self.niche_indices:
            return []

        index = self.niche_indices[niche_name]
        tags = self.niche_hashtag_maps[niche_name]

        query_emb = user_video_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)

        distances, indices = index.search(query_emb, min(top_k, len(tags)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "hashtag": tags[idx],
                "score": float(dist)
            })

        return results


if __name__ == "__main__":
    mock_embeddings = np.random.rand(10, 384)

    mock_metadata = [
        {'video_id': f'v{i}', 'hashtags': ['funny', 'cat', 'vlog']}
        for i in range(10)
    ]

    mock_niche_df = pd.DataFrame({
        'hashtag': ['funny', 'cat', 'vlog'],
        'niche_name': ['Entertainment', 'Pets', 'Life']
    })

    engine = NicheRetrievalEngine(embedding_dim=384)
    engine.compute_hashtag_representative_vectors(mock_embeddings, mock_metadata)
    engine.build_niche_faiss_indices(mock_niche_df)

    test_user_emb = np.random.rand(384)
    recommendations = engine.retrieve(test_user_emb, "Pets", top_k=5)

    print(recommendations)
