#!/usr/bin/env python3
"""
Convert Your Existing TikTok Dataset to Search System Format

Your dataset has 500k videos - this is amazing!
This script will:
1. Load your CSV
2. Map fields to our format
3. Optionally add AI summaries
4. Create search index
"""

import pandas as pd
import json
from pathlib import Path
from tiktok_search_system import VideoMetadata, TikTokEmbeddingSearch
from datetime import datetime
import sys


def load_dataset(csv_path: str, sample_size: int = None):
    """
    Load your TikTok dataset
    
    Args:
        csv_path: Path to your CSV file
        sample_size: If set, only load this many rows (for testing)
    """
    print("="*80)
    print("LOADING YOUR TIKTOK DATASET")
    print("="*80)
    print(f"Source: {csv_path}")
    
    try:
        # Load CSV
        if sample_size:
            print(f"Loading sample: {sample_size:,} videos...")
            df = pd.read_csv(csv_path, nrows=sample_size)
        else:
            print("Loading full dataset...")
            df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded {len(df):,} videos")
        print(f"Columns: {len(df.columns)}")
        print("="*80)
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)


def convert_to_metadata(df: pd.DataFrame) -> list:
    """
    Convert your dataset to VideoMetadata format
    
    Your fields → Our fields:
    - id → video_id
    - desc → caption (and initial description)
    - url → url
    - challenges → hashtags (needs parsing)
    - play_count → views
    - digg_count → likes
    - create_time → timestamp
    """
    print("\n" + "="*80)
    print("CONVERTING TO SEARCH FORMAT")
    print("="*80)
    
    videos = []
    
    for idx, row in df.iterrows():
        try:
            # Parse hashtags from challenges field
            # Format might be JSON string or comma-separated
            hashtags = []
            if pd.notna(row.get('challenges')):
                challenges_str = str(row['challenges'])
                # Try to parse as JSON first
                try:
                    import ast
                    challenges_list = ast.literal_eval(challenges_str)
                    if isinstance(challenges_list, list):
                        hashtags = [c.get('title', '') for c in challenges_list if isinstance(c, dict)]
                except:
                    # Fall back to simple parsing
                    hashtags = [challenges_str]
            
            # Create VideoMetadata object
            video = VideoMetadata(
                video_id=str(row.get('id', idx)),
                url=str(row.get('url', f'https://www.tiktok.com/video/{row.get("id", idx)}')),
                caption=str(row.get('desc', ''))[:500],  # Limit length
                description=str(row.get('desc', ''))[:500],  # Will enhance with AI later
                hashtags=hashtags[:10],  # Limit to 10 hashtags
                timestamp=str(row.get('create_time', datetime.now().isoformat())),
                views=int(row.get('play_count', 0)) if pd.notna(row.get('play_count')) else 0,
                likes=int(row.get('digg_count', 0)) if pd.notna(row.get('digg_count')) else 0
            )
            
            videos.append(video)
            
            # Progress
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,} videos...")
                
        except Exception as e:
            print(f"   ⚠️  Error at row {idx}: {e}")
            continue
    
    print(f"\n✓ Converted {len(videos):,} videos")
    print("="*80)
    
    return videos


def show_sample_data(videos: list, n: int = 3):
    """Show sample of converted data"""
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    
    for i, video in enumerate(videos[:n], 1):
        print(f"\n[{i}]")
        print(f"ID: {video.video_id}")
        print(f"Caption: {video.caption[:100]}...")
        print(f"Hashtags: {', '.join(video.hashtags[:5])}")
        print(f"Stats: 👁️  {video.views:,} views | ❤️  {video.likes:,} likes")
        print(f"URL: {video.url}")
        print("-"*80)


def get_dataset_stats(df: pd.DataFrame):
    """Show statistics about your dataset"""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nTotal videos: {len(df):,}")
    
    # View statistics
    if 'play_count' in df.columns:
        print(f"\nView Statistics:")
        print(f"  Total views: {df['play_count'].sum():,.0f}")
        print(f"  Average views: {df['play_count'].mean():,.0f}")
        print(f"  Median views: {df['play_count'].median():,.0f}")
        print(f"  Max views: {df['play_count'].max():,.0f}")
    
    # Like statistics
    if 'digg_count' in df.columns:
        print(f"\nLike Statistics:")
        print(f"  Total likes: {df['digg_count'].sum():,.0f}")
        print(f"  Average likes: {df['digg_count'].mean():,.0f}")
        print(f"  Median likes: {df['digg_count'].median():,.0f}")
    
    # Time range
    if 'create_time' in df.columns:
        print(f"\nTime Range:")
        try:
            df['create_time_parsed'] = pd.to_datetime(df['create_time'], errors='coerce')
            print(f"  Earliest: {df['create_time_parsed'].min()}")
            print(f"  Latest: {df['create_time_parsed'].max()}")
        except:
            print("  (Unable to parse dates)")
    
    print("="*80)


def enhance_with_ai(videos: list, batch_size: int = 1000, llm_provider: str = 'ollama'):
    """
    Optionally enhance descriptions with AI summaries
    
    Args:
        videos: List of VideoMetadata
        batch_size: Process this many at a time
        llm_provider: 'ollama', 'groq', etc.
    """
    print("\n" + "="*80)
    print("AI ENHANCEMENT (OPTIONAL)")
    print("="*80)
    print(f"You have {len(videos):,} videos")
    print(f"AI summaries would take approximately: {len(videos) * 2 / 3600:.1f} hours")
    print("\nOptions:")
    print("  1. Skip AI enhancement (use captions as-is)")
    print("  2. Enhance a sample (e.g., 1,000 videos)")
    print("  3. Enhance all videos")
    print("  4. Enhance top videos by engagement")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    if choice == '1':
        print("Skipping AI enhancement")
        return videos
    
    elif choice == '2':
        sample = int(input("How many videos to enhance? "))
        videos_to_enhance = videos[:sample]
    
    elif choice == '3':
        confirm = input(f"This will take ~{len(videos) * 2 / 3600:.1f} hours. Continue? (y/n): ")
        if confirm.lower() != 'y':
            return videos
        videos_to_enhance = videos
    
    elif choice == '4':
        # Sort by engagement and take top N
        sample = int(input("How many top videos to enhance? "))
        videos_sorted = sorted(videos, key=lambda v: v.views + v.likes, reverse=True)
        videos_to_enhance = videos_sorted[:sample]
    
    else:
        print("Invalid choice, skipping AI enhancement")
        return videos
    
    # Generate summaries
    try:
        from tiktok_auto_collection import FreeLLMSummarizer
        import time
        
        print(f"\nEnhancing {len(videos_to_enhance):,} videos with {llm_provider}...")
        summarizer = FreeLLMSummarizer(provider=llm_provider)
        
        enhanced_videos = set()
        
        for i, video in enumerate(videos_to_enhance, 1):
            try:
                summary = summarizer.generate_summary(
                    caption=video.caption,
                    hashtags=video.hashtags
                )
                video.description = summary
                enhanced_videos.add(video.video_id)
                
                if i % 100 == 0:
                    print(f"   Enhanced {i:,}/{len(videos_to_enhance):,} videos...")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"   ⚠️  Error at video {i}: {e}")
                continue
        
        print(f"\n✓ Enhanced {len(enhanced_videos):,} videos with AI summaries")
        
    except Exception as e:
        print(f"❌ Error during AI enhancement: {e}")
        print("Continuing without AI summaries...")
    
    return videos


def build_search_index(videos: list, 
                      index_name: str = 'tiktok_500k',
                      model_name: str = 'all-MiniLM-L6-v2',
                      batch_size: int = 32):
    """
    Build the search index
    
    For 500k videos:
    - all-MiniLM-L6-v2: ~30-45 min on CPU, 2GB RAM
    - all-mpnet-base-v2: ~60-90 min on CPU, 4GB RAM
    """
    print("\n" + "="*80)
    print("BUILDING SEARCH INDEX")
    print("="*80)
    print(f"Videos: {len(videos):,}")
    print(f"Model: {model_name}")
    print(f"Estimated time: {len(videos) * 0.005:.0f} minutes on CPU")
    print("="*80)
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Skipping index building")
        return None
    
    # Initialize search system
    search = TikTokEmbeddingSearch(
        model_name=model_name,
        data_dir='./tiktok_data'
    )
    
    # Build index
    print("\nGenerating embeddings...")
    search.add_videos(videos, batch_size=batch_size)
    
    # Save
    print(f"\nSaving as '{index_name}'...")
    search.save(name=index_name)
    
    # Statistics
    stats = search.get_statistics()
    print("\n" + "="*80)
    print("INDEX STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*80)
    
    return search


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert your 500k TikTok dataset to searchable index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 1000 videos first
  python convert_existing_dataset.py --csv your_data.csv --sample 1000
  
  # Build full index (all 500k videos)
  python convert_existing_dataset.py --csv your_data.csv --full
  
  # Build with AI enhancement for top 10k videos
  python convert_existing_dataset.py --csv your_data.csv --enhance-top 10000
  
  # Just show statistics
  python convert_existing_dataset.py --csv your_data.csv --stats-only
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to your TikTok CSV file')
    parser.add_argument('--sample', type=int, help='Test with N videos first')
    parser.add_argument('--full', action='store_true', help='Process all 500k videos')
    parser.add_argument('--enhance-top', type=int, help='Enhance top N videos with AI')
    parser.add_argument('--stats-only', action='store_true', help='Just show statistics')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', 
                       choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                       help='Embedding model')
    parser.add_argument('--index-name', default='tiktok_500k', help='Name for the index')
    
    args = parser.parse_args()
    
    # Determine how many videos to process
    if args.sample:
        sample_size = args.sample
    elif args.full:
        sample_size = None  # Load all
    elif args.stats_only:
        sample_size = None
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("DATASET SIZE SELECTION")
        print("="*80)
        print("You have 500k videos. Options:")
        print("  1. Test with 1,000 videos (~2 min)")
        print("  2. Test with 10,000 videos (~15 min)")
        print("  3. Process all 500k videos (~45 min)")
        
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == '1':
            sample_size = 1000
        elif choice == '2':
            sample_size = 10000
        elif choice == '3':
            sample_size = None
        else:
            print("Invalid choice, using 1000 for testing")
            sample_size = 1000
    
    # Load dataset
    df = load_dataset(args.csv, sample_size)
    
    # Show statistics
    get_dataset_stats(df)
    
    if args.stats_only:
        print("\n✓ Statistics complete")
        return
    
    # Convert to VideoMetadata
    videos = convert_to_metadata(df)
    
    # Show samples
    show_sample_data(videos)
    
    # Optional AI enhancement
    if args.enhance_top:
        print(f"\nEnhancing top {args.enhance_top:,} videos...")
        videos_sorted = sorted(videos, key=lambda v: v.views + v.likes, reverse=True)
        videos_to_enhance = videos_sorted[:args.enhance_top]
        
        from tiktok_auto_collection import FreeLLMSummarizer
        import time
        
        summarizer = FreeLLMSummarizer(provider='ollama')
        
        for i, video in enumerate(videos_to_enhance, 1):
            try:
                summary = summarizer.generate_summary(
                    caption=video.caption,
                    hashtags=video.hashtags
                )
                video.description = summary
                
                if i % 100 == 0:
                    print(f"   Enhanced {i:,}/{len(videos_to_enhance):,}...")
                
                time.sleep(0.5)
            except:
                continue
    
    # Build index
    search = build_search_index(
        videos, 
        index_name=args.index_name,
        model_name=args.model
    )
    
    if search:
        print("\n" + "="*80)
        print("🎉 SUCCESS!")
        print("="*80)
        print(f"✓ Processed {len(videos):,} videos")
        print(f"✓ Built search index: {args.index_name}")
        print("\nNext steps:")
        print("  python search_interface.py")
        print("\nTo search programmatically:")
        print(f"""
from tiktok_search_system import TikTokEmbeddingSearch

search = TikTokEmbeddingSearch()
search.load(name='{args.index_name}')

results = search.search("cooking recipes", top_k=10)
for r in results:
    print(r['caption'], r['similarity_score'])
        """)
        print("="*80)


if __name__ == "__main__":
    main()