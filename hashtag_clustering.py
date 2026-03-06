"""
Fixed: Cluster VIDEOS by hashtag similarity, then extract niches
With Unicode error handling
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import ast

print("Loading data...")
df = pd.read_csv('tiktok_500k.csv')
print(f"Loaded: {len(df):,} videos")

# Parse and clean hashtags
def parse_and_clean_hashtags(x):
    """Parse hashtags and remove problematic Unicode"""
    if pd.isna(x):
        return []
    try:
        # Parse
        if isinstance(x, list):
            parsed = [str(tag) for tag in x if tag]
        else:
            parsed = ast.literal_eval(str(x))
            if not isinstance(parsed, list):
                return []
        
        # Clean each hashtag
        cleaned = []
        for tag in parsed:
            try:
                # Remove # symbol
                tag = str(tag).strip().replace('#', '')
                
                # Handle Unicode - encode to UTF-8 and ignore errors
                tag = tag.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                
                # Remove emojis and special chars - keep only alphanumeric
                tag = ''.join(c for c in tag if c.isalnum() or c in ['_', '-'])
                
                # Convert to lowercase
                tag = tag.lower().strip()
                
                # Only keep if non-empty and reasonable length
                if tag and len(tag) > 1 and len(tag) < 50:
                    cleaned.append(tag)
            except:
                continue
        
        return cleaned
    except:
        return []

print("\nParsing and cleaning hashtags...")
df['hashtags'] = df['challenges'].apply(parse_and_clean_hashtags)

# Filter out videos with no hashtags
df = df[df['hashtags'].apply(len) > 0].reset_index(drop=True)
print(f"Videos with valid hashtags: {len(df):,}")

# Remove generic discovery tags
GENERIC = {
    'fyp', 'foryou', 'foryoupage', 'viral', 'trending', 'tiktok',
    'fy', 'fypviral', 'viralvideo', 'trend', 'fypage', 'foryourpage',
    'viraltiktok', 'xyzbca', 'explorepage', 'explore', 'capcut',
    'greenscreen', 'blowthisup', 'parati', 'paratii'
}

print(f"\nRemoving {len(GENERIC)} generic hashtags...")
df['hashtags'] = df['hashtags'].apply(lambda x: [h for h in x if h not in GENERIC])
df = df[df['hashtags'].apply(len) > 0].reset_index(drop=True)
print(f"Videos after filtering: {len(df):,}")

# Check hashtag statistics
all_tags = []
for tags in df['hashtags']:
    all_tags.extend(tags)

tag_counts = Counter(all_tags)
print(f"\nUnique hashtags: {len(tag_counts):,}")
print(f"Top 20 hashtags:")
for tag, count in tag_counts.most_common(20):
    print(f"  {tag}: {count:,}")

# Join hashtags into space-separated strings (safe now)
print("\nCreating hashtag text...")
df['hashtag_text'] = df['hashtags'].apply(lambda x: ' '.join(x))

# Verify no Unicode issues
print("Verifying Unicode safety...")
try:
    # Test encode all hashtag texts
    for i, text in enumerate(df['hashtag_text'].head(100)):
        text.encode('utf-8')
    print("✓ Unicode check passed")
except UnicodeEncodeError as e:
    print(f"❌ Unicode error found: {e}")
    print("Re-cleaning...")
    df['hashtag_text'] = df['hashtag_text'].apply(
        lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    )

# Create TF-IDF matrix of VIDEOS
print("\nBuilding TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    min_df=50,         # Hashtag must appear in 50+ videos
    max_df=0.5,        # But not more than 50% of videos
    max_features=1000, # Limit vocabulary
    token_pattern=r'\S+'  # Split on whitespace
)

video_matrix = vectorizer.fit_transform(df['hashtag_text'])
print(f"Video matrix shape: {video_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Cluster VIDEOS
print("\nClustering videos...")
n_clusters = 500
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters, 
    random_state=42, 
    batch_size=1000,
    max_iter=100
)
df['video_cluster'] = kmeans.fit_predict(video_matrix)

cluster_dist = df['video_cluster'].value_counts().sort_index()
print("\nVideo cluster distribution:")
print(cluster_dist)
print(f"\nStats:")
print(f"  Mean: {cluster_dist.mean():.0f}")
print(f"  Min: {cluster_dist.min()}")
print(f"  Max: {cluster_dist.max()}")

# Extract top hashtags from each VIDEO cluster
print("\nExtracting niche hashtags from clusters...")
cluster_niches = []

for cluster_id in range(n_clusters):
    cluster_videos = df[df['video_cluster'] == cluster_id]
    
    # Get all hashtags in this cluster
    all_cluster_tags = []
    for tags in cluster_videos['hashtags']:
        all_cluster_tags.extend(tags)
    
    # Count frequencies
    cluster_tag_counts = Counter(all_cluster_tags)
    top_tags = [tag for tag, count in cluster_tag_counts.most_common(20)]
    
    cluster_niches.append({
        'cluster_id': cluster_id,
        'num_videos': len(cluster_videos),
        'num_unique_hashtags': len(cluster_tag_counts),
        'top_10_hashtags': ', '.join(top_tags[:10]),
        'top_20_hashtags': ', '.join(top_tags[:20])
    })

niches_df = pd.DataFrame(cluster_niches)

print("\n" + "="*80)
print("VIDEO CLUSTERS (Niches)")
print("="*80)
print(niches_df[['cluster_id', 'num_videos', 'num_unique_hashtags', 'top_10_hashtags']].to_string(index=False))

# Save results
print("\nSaving results...")
niches_df.to_csv('video_clusters.csv', index=False)
df[['video_cluster', 'hashtags']].to_csv('video_cluster_assignments.csv', index=False)

print("\n✓ Saved:")
print("  - video_clusters.csv (cluster summaries)")
print("  - video_cluster_assignments.csv (video assignments)")

print("\n✓ Done!")