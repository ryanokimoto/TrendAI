"""
TikTok Hashtag Clustering with LLM-based Niche Naming
Uses co-occurrence clustering and Ollama (free open-source LLM)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ast
import requests
import json
from collections import Counter

print("Loading TikTok dataset...")
df = pd.read_csv('tiktok_500k.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Parse the 'challenges' field (stored as string representation of list)
print("\nParsing hashtags from 'challenges' field...")

def parse_hashtags(x):
    """Parse hashtags from string representation of list"""
    if pd.isna(x):
        return []
    try:
        # If it's already a list
        if isinstance(x, list):
            return [str(tag).strip() for tag in x if tag]
        # If it's a string representation of a list
        parsed = ast.literal_eval(x)
        return [str(tag).strip() for tag in parsed if tag]
    except:
        # If parsing fails, try splitting by common delimiters
        if isinstance(x, str):
            return [tag.strip() for tag in x.replace('[', '').replace(']', '').replace("'", '').split(',') if tag.strip()]
        return []

df['hashtags'] = df['challenges'].apply(parse_hashtags)

# Filter out videos with no hashtags
df = df[df['hashtags'].apply(len) > 0].reset_index(drop=True)
print(f"Videos with hashtags: {len(df)}")

# Clean hashtags to remove problematic unicode characters
def clean_hashtag(tag):
    """Remove emojis and problematic unicode characters"""
    try:
        # Encode to ASCII, ignore errors
        return tag.encode('ascii', 'ignore').decode('ascii')
    except:
        return ''

df['hashtags'] = df['hashtags'].apply(
    lambda tags: [clean_hashtag(tag) for tag in tags if clean_hashtag(tag)]
)

# Filter out videos with no hashtags after cleaning
df = df[df['hashtags'].apply(len) > 0].reset_index(drop=True)
print(f"Videos after cleaning: {len(df)}")

# Get hashtag frequency - convert to object dtype to avoid PyArrow issues
all_hashtags = df['hashtags'].astype('object').explode()
hashtag_counts = all_hashtags.value_counts()
print(f"\nTotal unique hashtags: {len(hashtag_counts)}")
print(f"Top 10 hashtags:\n{hashtag_counts.head(10)}")

# Filter hashtags - keep those appearing in at least 10 videos
min_frequency = 10
popular_hashtags = set(hashtag_counts[hashtag_counts >= min_frequency].index)
print(f"\nHashtags appearing in ≥{min_frequency} videos: {len(popular_hashtags)}")

# Filter dataset to only include popular hashtags
df['hashtags_filtered'] = df['hashtags'].apply(lambda x: [h for h in x if h in popular_hashtags])
df = df[df['hashtags_filtered'].apply(len) > 0].reset_index(drop=True)
print(f"Videos after filtering: {len(df)}")

# Create co-occurrence matrix
print("\nBuilding co-occurrence matrix...")
vectorizer = CountVectorizer(
    tokenizer=lambda x: x, 
    lowercase=False, 
    binary=True,
    min_df=min_frequency
)

hashtag_matrix = vectorizer.fit_transform(df['hashtags_filtered'])
hashtag_names = vectorizer.get_feature_names_out()

print(f"Matrix shape: {hashtag_matrix.shape}")
print(f"Number of hashtags in matrix: {len(hashtag_names)}")

# Dimensionality reduction
print("\nReducing dimensionality with SVD...")
n_components = min(100, min(hashtag_matrix.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
hashtag_embeddings = svd.fit_transform(hashtag_matrix.T)

print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

# Determine optimal number of clusters using elbow method
print("\nFinding optimal number of clusters...")
inertias = []
silhouette_scores = []
K_range = range(10, 51, 5)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(hashtag_embeddings)
    inertias.append(kmeans.inertia_)
    if k <= 30:  # Silhouette score is expensive for large k
        score = silhouette_score(hashtag_embeddings, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"k={k}: inertia={kmeans.inertia_:.0f}, silhouette={score:.3f}")
    else:
        print(f"k={k}: inertia={kmeans.inertia_:.0f}")

# Use a reasonable default number of clusters
n_clusters = 30
print(f"\nUsing {n_clusters} clusters")

# Final clustering
print("\nPerforming final clustering...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
clusters = kmeans.fit_predict(hashtag_embeddings)

# Create hashtag-cluster mapping
hashtag_clusters = pd.DataFrame({
    'hashtag': hashtag_names,
    'cluster': clusters
})

# Add frequency information
hashtag_clusters['frequency'] = hashtag_clusters['hashtag'].map(hashtag_counts)
hashtag_clusters = hashtag_clusters.sort_values(['cluster', 'frequency'], ascending=[True, False])

print(f"\nCluster distribution:")
print(hashtag_clusters['cluster'].value_counts().sort_index())

# Function to call Ollama LLM for niche naming
def get_niche_name_from_llm(top_hashtags, cluster_id):
    """Use Ollama to generate a niche name from top hashtags"""
    try:
        prompt = f"""Analyze these TikTok hashtags that frequently appear together:

{', '.join(top_hashtags)}

Based on these hashtags, what is the main topic/niche they represent? 
Provide ONLY a concise 2-4 word category name (e.g., "Beauty & Makeup", "Gaming Content", "Fitness Tips").
Do not explain, just give the category name."""

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 20
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            niche_name = result['response'].strip()
            # Clean up the response
            niche_name = niche_name.replace('"', '').replace("'", '').strip()
            # Take only first line if multiple lines
            niche_name = niche_name.split('\n')[0].strip()
            return niche_name
        else:
            print(f"  ⚠ API error for cluster {cluster_id}: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"  ⚠ Cannot connect to Ollama. Make sure it's running: 'ollama serve'")
        return None
    except Exception as e:
        print(f"  ⚠ Error for cluster {cluster_id}: {str(e)}")
        return None

# Generate niche names
print("\n" + "="*60)
print("GENERATING NICHE NAMES WITH LLM")
print("="*60)
print("\nNote: Make sure Ollama is installed and running!")
print("Install: curl -fsSL https://ollama.com/install.sh | sh")
print("Run: ollama run llama3.2")
print("="*60 + "\n")

cluster_info = []

for cluster_id in sorted(hashtag_clusters['cluster'].unique()):
    cluster_tags = hashtag_clusters[hashtag_clusters['cluster'] == cluster_id]
    
    # Get top hashtags by frequency
    top_tags = cluster_tags.nlargest(20, 'frequency')['hashtag'].tolist()
    
    print(f"\nCluster {cluster_id} ({len(cluster_tags)} hashtags)")
    print(f"  Top hashtags: {', '.join(top_tags[:10])}")
    
    # Try to get LLM-generated name
    llm_name = get_niche_name_from_llm(top_tags[:15], cluster_id)
    
    # Fallback to most frequent hashtag if LLM fails
    fallback_name = top_tags[0]
    
    final_name = llm_name if llm_name else fallback_name
    name_source = "LLM" if llm_name else "Fallback"
    
    print(f"  Niche name: '{final_name}' ({name_source})")
    
    cluster_info.append({
        'cluster_id': cluster_id,
        'niche_name': final_name,
        'name_source': name_source,
        'num_hashtags': len(cluster_tags),
        'top_5_hashtags': ', '.join(top_tags[:5]),
        'top_20_hashtags': ', '.join(top_tags[:20])
    })

# Create summary DataFrame
summary_df = pd.DataFrame(cluster_info)

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save cluster summary
summary_df.to_csv('cluster_summary.csv', index=False)
print("\n✓ Saved cluster summary to: cluster_summary.csv")

# Save full hashtag-cluster mapping
hashtag_clusters_with_names = hashtag_clusters.merge(
    summary_df[['cluster_id', 'niche_name']], 
    left_on='cluster', 
    right_on='cluster_id',
    how='left'
)
hashtag_clusters_with_names = hashtag_clusters_with_names[['hashtag', 'cluster', 'niche_name', 'frequency']]
hashtag_clusters_with_names.to_csv('hashtag_to_niche.csv', index=False)
print("✓ Saved hashtag-to-niche mapping to: hashtag_to_niche.csv")

# Print summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nTotal videos analyzed: {len(df):,}")
print(f"Unique hashtags clustered: {len(hashtag_names):,}")
print(f"Number of niches identified: {n_clusters}")
print(f"\nLLM-generated names: {sum(summary_df['name_source'] == 'LLM')}")
print(f"Fallback names: {sum(summary_df['name_source'] == 'Fallback')}")

print("\n" + "="*60)
print("TOP 10 NICHES BY SIZE")
print("="*60)
print(summary_df.nlargest(10, 'num_hashtags')[['niche_name', 'num_hashtags', 'top_5_hashtags']].to_string(index=False))

print("\n✓ Done! Check cluster_summary.csv for review and manual refinement.")