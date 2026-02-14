"""
TikTok Video Semantic Search System
Handles 5k+ videos with efficient embedding-based search
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Install: pip install sentence-transformers faiss-cpu --break-system-packages
from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class VideoMetadata:
    """Stores metadata for each video"""
    video_id: str
    url: str
    description: str
    caption: str
    hashtags: List[str]
    timestamp: str
    views: Optional[int] = None
    likes: Optional[int] = None
    
    def get_searchable_text(self) -> str:
        """Combine all text fields for embedding"""
        hashtag_text = " ".join(self.hashtags)
        return f"{self.caption}. {self.description}. {hashtag_text}"


class TikTokEmbeddingSearch:
    """Main search system using sentence embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', data_dir: str = './tiktok_data'):
        """
        Initialize the search system
        
        Args:
            model_name: HuggingFace model name for embeddings
                       Options: 'all-MiniLM-L6-v2' (fast, 384 dim)
                               'all-mpnet-base-v2' (better quality, 768 dim)
            data_dir: Directory to store data and index
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.videos: List[VideoMetadata] = []
        self.index: Optional[faiss.Index] = None
        
    def add_videos(self, videos: List[VideoMetadata], batch_size: int = 32):
        """
        Add videos and create embeddings
        
        Args:
            videos: List of VideoMetadata objects
            batch_size: Batch size for embedding generation
        """
        print(f"\nAdding {len(videos)} videos...")
        
        # Add to our list
        start_idx = len(self.videos)
        self.videos.extend(videos)
        
        # Generate embeddings in batches
        texts = [v.get_searchable_text() for v in videos]
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create or update FAISS index
        if self.index is None:
            # Create new index (using L2 distance, normalize for cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Added {len(videos)} videos (total: {len(self.videos)})")
        
    def search(self, query: str, top_k: int = 10, min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for videos similar to query
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of results with video metadata and similarity scores
        """
        if self.index is None or len(self.videos) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= min_similarity:
                video = self.videos[idx]
                results.append({
                    'video_id': video.video_id,
                    'url': video.url,
                    'caption': video.caption,
                    'description': video.description,
                    'hashtags': video.hashtags,
                    'similarity_score': float(similarity),
                    'views': video.views,
                    'likes': video.likes
                })
        
        return results
    
    def save(self, name: str = 'tiktok_search'):
        """Save index and metadata to disk"""
        print(f"\nSaving to {self.data_dir / name}...")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.data_dir / f"{name}.index"))
        
        # Save video metadata
        videos_dict = [asdict(v) for v in self.videos]
        with open(self.data_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(videos_dict, f, indent=2)
        
        # Save config
        config = {
            'model_name': self.model.get_sentence_embedding_dimension(),
            'num_videos': len(self.videos),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.data_dir / f"{name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Saved successfully")
    
    def load(self, name: str = 'tiktok_search'):
        """Load index and metadata from disk"""
        print(f"\nLoading from {self.data_dir / name}...")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.data_dir / f"{name}.index"))
        
        # Load video metadata
        with open(self.data_dir / f"{name}_metadata.json", 'r') as f:
            videos_dict = json.load(f)
            self.videos = [VideoMetadata(**v) for v in videos_dict]
        
        print(f"✓ Loaded {len(self.videos)} videos")
        
    def get_statistics(self) -> Dict:
        """Get statistics about the index"""
        return {
            'total_videos': len(self.videos),
            'embedding_dimension': self.embedding_dim,
            'model': type(self.model).__name__,
            'index_size_mb': self.index.ntotal * self.embedding_dim * 4 / (1024**2) if self.index else 0
        }


def create_sample_videos(n: int = 100) -> List[VideoMetadata]:
    """
    Create sample video data for testing
    Replace this with your actual data collection
    """
    import random
    
    categories = [
        ("cooking", ["pasta", "baking", "meal prep", "recipes"]),
        ("fitness", ["workout", "yoga", "running", "gym"]),
        ("comedy", ["funny", "prank", "sketch", "meme"]),
        ("education", ["tutorial", "howto", "learning", "tips"]),
        ("pets", ["cats", "dogs", "cute animals", "pet tricks"]),
        ("travel", ["beach", "city tour", "adventure", "vacation"]),
        ("dance", ["choreography", "trending dance", "ballet", "hip hop"]),
        ("art", ["drawing", "painting", "crafts", "DIY"])
    ]
    
    videos = []
    for i in range(n):
        category, tags = random.choice(categories)
        video = VideoMetadata(
            video_id=f"video_{i:05d}",
            url=f"https://tiktok.com/@user/video/{i}",
            caption=f"Check out this {category} video! #{random.choice(tags)}",
            description=f"A {category} video showing {random.choice(tags)}",
            hashtags=[category] + random.sample(tags, 2),
            timestamp=datetime.now().isoformat(),
            views=random.randint(100, 1000000),
            likes=random.randint(10, 50000)
        )
        videos.append(video)
    
    return videos


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("TikTok Video Semantic Search System")
    print("="*60)
    
    # Initialize system
    search_system = TikTokEmbeddingSearch(
        model_name='all-MiniLM-L6-v2',  # Fast and efficient
        data_dir='./tiktok_data'
    )
    
    # Generate sample data (replace with your actual data collection)
    print("\n[1] Generating sample videos...")
    sample_videos = create_sample_videos(n=500)  # Start with 500 for testing
    
    # Add videos to system
    print("\n[2] Creating embeddings...")
    search_system.add_videos(sample_videos, batch_size=32)
    
    # Save the index
    print("\n[3] Saving index...")
    search_system.save()
    
    # Test searches
    print("\n[4] Testing searches...")
    print("\n" + "="*60)
    
    test_queries = [
        "funny cat videos",
        "how to cook pasta",
        "workout routines",
        "travel to beaches",
        "dance tutorials"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        results = search_system.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['similarity_score']:.3f}] {result['caption']}")
            print(f"   Description: {result['description']}")
            print(f"   Hashtags: {', '.join(result['hashtags'])}")
            print(f"   Views: {result['views']:,} | Likes: {result['likes']:,}")
    
    # Show statistics
    print("\n" + "="*60)
    print("System Statistics:")
    print("="*60)
    stats = search_system.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")