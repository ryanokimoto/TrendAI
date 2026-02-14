"""
Automated TikTok Data Collection Pipeline
Scrapes videos + generates AI summaries using free LLMs
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

from tiktok_search_system import VideoMetadata


# ============================================================================
# FREE LLM OPTIONS
# ============================================================================

class FreeLLMSummarizer:
    """
    Wrapper for various free LLM APIs
    Choose the one that works best for you
    """
    
    def __init__(self, provider: str = 'ollama'):
        """
        Initialize LLM summarizer
        
        Providers:
            'ollama' - Run locally (best for privacy, no limits)
            'groq' - Free API, very fast
            'together' - Free tier available
            'huggingface' - Free inference API
        """
        self.provider = provider
        self.setup_provider()
    
    def setup_provider(self):
        """Setup based on chosen provider"""
        
        if self.provider == 'ollama':
            # Ollama runs locally - completely free, no API needed
            # Install: https://ollama.ai
            # Run: ollama pull llama3.2
            self.endpoint = "http://localhost:11434/api/generate"
            self.model = "llama3.2"  # or "mistral", "phi3", etc.
            print("Using Ollama (local) - Make sure Ollama is running!")
            
        elif self.provider == 'groq':
            # Groq - Free tier: https://console.groq.com
            # Very fast inference
            import os
            self.api_key = os.getenv('GROQ_API_KEY')
            self.model = "llama-3.1-8b-instant"
            print("Using Groq API - Make sure GROQ_API_KEY is set")
            
        elif self.provider == 'together':
            # Together AI - Free tier available
            self.api_key = os.getenv('TOGETHER_API_KEY')
            self.model = "meta-llama/Llama-3-8b-chat-hf"
            print("Using Together AI - Make sure TOGETHER_API_KEY is set")
            
        elif self.provider == 'huggingface':
            # HuggingFace Inference API - Free
            self.api_key = os.getenv('HF_API_KEY')
            self.model = "meta-llama/Meta-Llama-3-8B-Instruct"
            print("Using HuggingFace - Make sure HF_API_KEY is set")
    
    def summarize_ollama(self, caption: str, hashtags: List[str]) -> str:
        """Generate summary using local Ollama"""
        import requests
        
        hashtag_text = ", ".join(hashtags)
        prompt = f"""Summarize this TikTok video in 1-2 concise sentences. Focus on what the video is about.

Caption: {caption}
Hashtags: {hashtag_text}

Summary:"""
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 100
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"{caption}"  # Fallback to caption
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return f"{caption}"
    
    def summarize_groq(self, caption: str, hashtags: List[str]) -> str:
        """Generate summary using Groq API"""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            hashtag_text = ", ".join(hashtags)
            
            prompt = f"""Summarize this TikTok video in 1-2 concise sentences.

Caption: {caption}
Hashtags: {hashtag_text}

Summary:"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq error: {e}")
            return f"{caption}"
    
    def summarize_together(self, caption: str, hashtags: List[str]) -> str:
        """Generate summary using Together AI"""
        import requests
        
        hashtag_text = ", ".join(hashtags)
        prompt = f"""Summarize this TikTok video in 1-2 concise sentences.

Caption: {caption}
Hashtags: {hashtag_text}

Summary:"""
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 100
                }
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"{caption}"
                
        except Exception as e:
            print(f"Together AI error: {e}")
            return f"{caption}"
    
    def summarize_huggingface(self, caption: str, hashtags: List[str]) -> str:
        """Generate summary using HuggingFace Inference API"""
        import requests
        
        hashtag_text = ", ".join(hashtags)
        prompt = f"Summarize in 1-2 sentences: {caption} (tags: {hashtag_text})"
        
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 100}}
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', caption).strip()
            return f"{caption}"
            
        except Exception as e:
            print(f"HuggingFace error: {e}")
            return f"{caption}"
    
    def generate_summary(self, caption: str, hashtags: List[str]) -> str:
        """
        Generate summary using configured provider
        
        Args:
            caption: Video caption
            hashtags: List of hashtags
            
        Returns:
            AI-generated summary
        """
        if self.provider == 'ollama':
            return self.summarize_ollama(caption, hashtags)
        elif self.provider == 'groq':
            return self.summarize_groq(caption, hashtags)
        elif self.provider == 'together':
            return self.summarize_together(caption, hashtags)
        elif self.provider == 'huggingface':
            return self.summarize_huggingface(caption, hashtags)
        else:
            return f"{caption}"  # Fallback


# ============================================================================
# AUTOMATED COLLECTION PIPELINE
# ============================================================================

class TikTokCollectionPipeline:
    """
    Complete pipeline: Scrape → Summarize → Save
    """
    
    def __init__(self, 
                 llm_provider: str = 'ollama',
                 output_dir: str = './collected_data',
                 rate_limit_delay: float = 2.0):
        """
        Initialize collection pipeline
        
        Args:
            llm_provider: Which LLM to use ('ollama', 'groq', 'together', 'huggingface')
            output_dir: Where to save collected data
            rate_limit_delay: Seconds to wait between videos (avoid rate limits)
        """
        self.summarizer = FreeLLMSummarizer(provider=llm_provider)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        
        self.collected_videos: List[VideoMetadata] = []
    
    async def scrape_tiktok(self, hashtags: List[str], videos_per_tag: int = 100) -> List[Dict]:
        """
        Scrape TikTok for videos by hashtags
        
        Args:
            hashtags: List of hashtags to scrape
            videos_per_tag: Number of videos to collect per hashtag
            
        Returns:
            List of raw video data
        """
        print(f"\n{'='*80}")
        print(f"SCRAPING TIKTOK VIDEOS")
        print(f"{'='*80}")
        print(f"Hashtags: {', '.join(hashtags)}")
        print(f"Videos per tag: {videos_per_tag}")
        print(f"{'='*80}\n")
        
        try:
            from TikTokApi import TikTokApi
        except ImportError:
            print("❌ TikTokApi not installed!")
            print("Install: pip install TikTokApi playwright --break-system-packages")
            print("Then run: playwright install")
            return []
        
        all_videos = []
        
        async with TikTokApi() as api:
            for hashtag in hashtags:
                print(f"\n📱 Scraping #{hashtag}...")
                
                try:
                    tag = api.hashtag(name=hashtag)
                    count = 0
                    
                    async for video in tag.videos(count=videos_per_tag):
                        try:
                            video_data = {
                                'id': video.id,
                                'author': video.author.unique_id if video.author else 'unknown',
                                'caption': video.desc or '',
                                'hashtags': [hashtag] + [t.get('name', '') for t in (video.challenges or [])],
                                'views': video.stats.get('playCount', 0) if video.stats else 0,
                                'likes': video.stats.get('diggCount', 0) if video.stats else 0,
                                'shares': video.stats.get('shareCount', 0) if video.stats else 0,
                                'comments': video.stats.get('commentCount', 0) if video.stats else 0,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            all_videos.append(video_data)
                            count += 1
                            
                            if count % 10 == 0:
                                print(f"   Collected {count}/{videos_per_tag} videos...")
                            
                        except Exception as e:
                            print(f"   ⚠️  Error processing video: {e}")
                            continue
                    
                    print(f"   ✓ Collected {count} videos for #{hashtag}")
                    
                    # Rate limiting between hashtags
                    await asyncio.sleep(self.rate_limit_delay * 2)
                    
                except Exception as e:
                    print(f"   ❌ Error scraping #{hashtag}: {e}")
                    continue
        
        print(f"\n{'='*80}")
        print(f"✓ Total videos scraped: {len(all_videos)}")
        print(f"{'='*80}\n")
        
        return all_videos
    
    def generate_summaries(self, videos: List[Dict], batch_size: int = 10) -> List[VideoMetadata]:
        """
        Generate AI summaries for scraped videos
        
        Args:
            videos: Raw video data from scraping
            batch_size: Process this many before showing progress
            
        Returns:
            List of VideoMetadata with AI summaries
        """
        print(f"\n{'='*80}")
        print(f"GENERATING AI SUMMARIES")
        print(f"{'='*80}")
        print(f"Provider: {self.summarizer.provider}")
        print(f"Videos to process: {len(videos)}")
        print(f"{'='*80}\n")
        
        processed = []
        
        for i, video in enumerate(videos, 1):
            try:
                # Generate summary
                summary = self.summarizer.generate_summary(
                    caption=video['caption'],
                    hashtags=video['hashtags']
                )
                
                # Create VideoMetadata object
                metadata = VideoMetadata(
                    video_id=video['id'],
                    url=f"https://www.tiktok.com/@{video['author']}/video/{video['id']}",
                    caption=video['caption'],
                    description=summary,  # AI-generated summary goes here!
                    hashtags=video['hashtags'],
                    timestamp=video['timestamp'],
                    views=video['views'],
                    likes=video['likes']
                )
                
                processed.append(metadata)
                
                # Progress update
                if i % batch_size == 0:
                    print(f"   Processed {i}/{len(videos)} videos...")
                
                # Rate limiting to avoid overwhelming the LLM
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"   ⚠️  Error processing video {video['id']}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"✓ Generated summaries for {len(processed)} videos")
        print(f"{'='*80}\n")
        
        return processed
    
    def save_data(self, videos: List[VideoMetadata], filename: str = None):
        """Save collected and summarized data"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tiktok_collection_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to dict for JSON serialization
        data = {
            'collection_date': datetime.now().isoformat(),
            'total_videos': len(videos),
            'llm_provider': self.summarizer.provider,
            'videos': [
                {
                    'video_id': v.video_id,
                    'url': v.url,
                    'caption': v.caption,
                    'ai_summary': v.description,  # AI-generated!
                    'hashtags': v.hashtags,
                    'timestamp': v.timestamp,
                    'views': v.views,
                    'likes': v.likes
                }
                for v in videos
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved to: {output_path}")
        
        # Also save as CSV for easy import
        csv_path = output_path.with_suffix('.csv')
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'url', 'caption', 'description', 'hashtags', 'views', 'likes'])
            
            for v in videos:
                writer.writerow([
                    v.video_id,
                    v.url,
                    v.caption,
                    v.description,
                    ','.join(v.hashtags),
                    v.views,
                    v.likes
                ])
        
        print(f"✓ Also saved CSV: {csv_path}")
        
        return output_path
    
    async def run_collection(self, 
                            hashtags: List[str], 
                            videos_per_tag: int = 100,
                            save_filename: str = None) -> List[VideoMetadata]:
        """
        Complete pipeline: Scrape → Summarize → Save
        
        Args:
            hashtags: Hashtags to scrape
            videos_per_tag: Videos per hashtag
            save_filename: Optional custom filename
            
        Returns:
            List of processed VideoMetadata
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATED TIKTOK COLLECTION PIPELINE")
        print(f"{'='*80}\n")
        
        # Step 1: Scrape
        raw_videos = await self.scrape_tiktok(hashtags, videos_per_tag)
        
        if not raw_videos:
            print("❌ No videos collected!")
            return []
        
        # Step 2: Generate summaries
        processed_videos = self.generate_summaries(raw_videos)
        
        # Step 3: Save
        self.save_data(processed_videos, save_filename)
        
        # Store in pipeline
        self.collected_videos.extend(processed_videos)
        
        print(f"\n{'='*80}")
        print(f"✓ PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total videos collected: {len(processed_videos)}")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*80}\n")
        
        return processed_videos


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_collection():
    """Example: Collect 500 videos with AI summaries"""
    
    # Initialize pipeline with Ollama (free, local)
    pipeline = TikTokCollectionPipeline(
        llm_provider='ollama',  # or 'groq', 'together', 'huggingface'
        output_dir='./collected_data',
        rate_limit_delay=1.0  # 1 second between videos
    )
    
    # Define hashtags to scrape
    hashtags = [
        'cooking',
        'fitness', 
        'comedy',
        'education',
        'travel'
    ]
    
    # Run collection (100 videos per tag = 500 total)
    videos = await pipeline.run_collection(
        hashtags=hashtags,
        videos_per_tag=100
    )
    
    # Show sample results
    print("\n📊 Sample Results:")
    print("="*80)
    for i, video in enumerate(videos[:3], 1):
        print(f"\n[{i}]")
        print(f"Caption: {video.caption}")
        print(f"AI Summary: {video.description}")
        print(f"Hashtags: {', '.join(video.hashtags)}")
        print(f"Engagement: 👁️ {video.views:,} | ❤️ {video.likes:,}")
        print("-"*80)


async def example_batch_collection_5k():
    """Example: Collect 5k videos in batches"""
    
    pipeline = TikTokCollectionPipeline(llm_provider='ollama')
    
    # Broader set of hashtags
    hashtags = [
        'cooking', 'fitness', 'comedy', 'education', 'travel',
        'pets', 'dance', 'art', 'diy', 'music',
        'fashion', 'beauty', 'gaming', 'tech', 'life'
    ]
    
    # Collect ~330 videos per tag = ~5k total
    videos = await pipeline.run_collection(
        hashtags=hashtags,
        videos_per_tag=330
    )
    
    print(f"\n✓ Collected {len(videos)} videos for search index!")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  TikTok Automated Collection Pipeline with Free LLM Summaries ║
    ╚════════════════════════════════════════════════════════════════╝
    
    FREE LLM OPTIONS:
    ─────────────────
    1. Ollama (Recommended) - Run locally, completely free
       • Install: https://ollama.ai
       • Run: ollama pull llama3.2
       • Best for: Privacy, no limits, offline
    
    2. Groq - Free API, very fast
       • Sign up: https://console.groq.com
       • Set: export GROQ_API_KEY=your_key
       • Best for: Speed, cloud-based
    
    3. Together AI - Free tier
       • Sign up: https://together.ai
       • Set: export TOGETHER_API_KEY=your_key
    
    4. HuggingFace - Free inference
       • Sign up: https://huggingface.co
       • Set: export HF_API_KEY=your_key
    
    USAGE:
    ──────
    # For 500 videos
    python tiktok_auto_collection.py
    
    # Then build search index
    python build_index.py --csv collected_data/tiktok_collection_*.csv
    """)
    
    # Run example collection
    print("\nStarting collection pipeline...")
    asyncio.run(example_collection())