#!/usr/bin/env python3
"""
Vision AI Video Descriptions
Downloads videos and uses vision models to describe content
"""

import pandas as pd
import sys
import subprocess
import time
import os
from pathlib import Path
from typing import Optional
import json


class VideoVisionDescriber:
    """Generate descriptions using vision AI on video frames"""
    
    def __init__(self, 
                 vision_provider: str = 'llava',
                 download_dir: str = './downloaded_videos',
                 keep_videos: bool = False):
        """
        Args:
            vision_provider: 'llava' (free, local) or 'gpt4v' (paid, cloud)
            download_dir: Where to save downloaded videos
            keep_videos: Keep videos after processing (uses lots of storage)
        """
        self.vision_provider = vision_provider
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.keep_videos = keep_videos
        
        self.setup_vision_model()
    
    def setup_vision_model(self):
        """Setup vision model based on provider"""
        
        if self.vision_provider == 'llava':
            print("Checking LLaVA (Ollama vision model)...")
            
            try:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'llava' not in result.stdout.lower():
                    print("\n❌ LLaVA not installed!")
                    print("\nInstall it:")
                    print("  ollama pull llava")
                    print("\nOr for better quality:")
                    print("  ollama pull llava:13b")
                    sys.exit(1)
                else:
                    print("✓ LLaVA ready")
                    
            except Exception as e:
                print(f"\n❌ Ollama not available: {e}")
                print("\nInstall from https://ollama.ai")
                sys.exit(1)
        
        elif self.vision_provider == 'gpt4v':
            print("Checking GPT-4 Vision...")
            
            try:
                import openai
                if not os.getenv('OPENAI_API_KEY'):
                    print("❌ OPENAI_API_KEY not set!")
                    print("Set it: export OPENAI_API_KEY=sk-...")
                    sys.exit(1)
                print("✓ OpenAI API key found")
            except ImportError:
                print("❌ openai package not installed!")
                print("Install: pip install openai --break-system-packages")
                sys.exit(1)
    
    def download_video(self, url: str, video_id: str) -> Optional[str]:
        """
        Download video using yt-dlp
        
        Returns:
            Path to downloaded video or None if failed
        """
        output_path = self.download_dir / f"{video_id}.mp4"
        
        # Skip if already downloaded
        if output_path.exists():
            return str(output_path)
        
        try:
            # Use yt-dlp to download
            subprocess.run([
                'yt-dlp',
                '-f', 'worst',  # Download lowest quality (faster, smaller)
                '-o', str(output_path),
                '--no-playlist',
                '--quiet',
                url
            ], check=True, timeout=60)
            
            if output_path.exists():
                return str(output_path)
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠️  Download failed: {e}")
            return None
    
    def extract_frame(self, video_path: str) -> Optional[str]:
        """
        Extract middle frame from video
        
        Returns:
            Path to frame image
        """
        try:
            import cv2
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Get middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                return None
                
            middle_frame = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Save frame
            frame_path = video_path.replace('.mp4', '_frame.jpg')
            cv2.imwrite(frame_path, frame)
            
            return frame_path
            
        except Exception as e:
            print(f"  ⚠️  Frame extraction failed: {e}")
            return None
    
    def describe_with_llava(self, frame_path: str) -> str:
        """Generate description using LLaVA"""
        
        prompt = """Describe this TikTok video in 2-3 concise sentences. Focus on:
- What is the main subject/person doing?
- What is the setting/environment?
- Any notable objects or actions?

Keep it factual and brief."""
        
        try:
            result = subprocess.run(
                ['ollama', 'run', 'llava', prompt, frame_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            description = result.stdout.strip()
            
            # Clean up output
            if description and len(description) > 20:
                return description
            else:
                return "Unable to generate description"
                
        except Exception as e:
            return f"Error: {e}"
    
    def describe_with_gpt4v(self, frame_path: str) -> str:
        """Generate description using GPT-4 Vision"""
        
        try:
            import openai
            import base64
            
            # Read and encode frame
            with open(frame_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Call GPT-4 Vision
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this TikTok video in 2-3 sentences. What's happening? Who/what is in it? What's the setting?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {e}"
    
    def process_video(self, url: str, video_id: str) -> str:
        """
        Download video, extract frame, and generate description
        
        Returns:
            Description text
        """
        # Download video
        video_path = self.download_video(url, video_id)
        if not video_path:
            return "Video download failed"
        
        # Extract frame
        frame_path = self.extract_frame(video_path)
        if not frame_path:
            return "Frame extraction failed"
        
        # Generate description
        if self.vision_provider == 'llava':
            description = self.describe_with_llava(frame_path)
        elif self.vision_provider == 'gpt4v':
            description = self.describe_with_gpt4v(frame_path)
        else:
            description = "Unknown vision provider"
        
        # Cleanup
        if not self.keep_videos:
            try:
                os.remove(video_path)
                os.remove(frame_path)
            except:
                pass
        
        return description
    
    def process_dataset(self, 
                       df: pd.DataFrame,
                       url_column: str = 'url',
                       id_column: str = 'id',
                       max_videos: Optional[int] = None) -> pd.DataFrame:
        """
        Process dataset and add vision-based descriptions
        
        Args:
            df: DataFrame with video data
            url_column: Column name with video URLs
            id_column: Column name with video IDs
            max_videos: Process at most this many (None = all)
        """
        
        print("\n" + "="*80)
        print("GENERATING VISION-BASED DESCRIPTIONS")
        print("="*80)
        print(f"Videos: {len(df) if not max_videos else min(len(df), max_videos):,}")
        print(f"Vision Model: {self.vision_provider}")
        print(f"Keep Videos: {self.keep_videos}")
        print("="*80 + "\n")
        
        print("⚠️  This will download videos and take significant time!")
        print(f"Estimated time: {(len(df) if not max_videos else max_videos) * 10 / 3600:.1f} hours")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            sys.exit(0)
        
        descriptions = []
        successful = 0
        
        # Process videos
        videos_to_process = df.head(max_videos) if max_videos else df
        
        for idx, row in videos_to_process.iterrows():
            video_id = str(row[id_column])
            url = str(row[url_column])
            
            print(f"\n[{idx + 1}/{len(videos_to_process)}] Processing {video_id}...")
            
            description = self.process_video(url, video_id)
            descriptions.append(description)
            
            if 'failed' not in description.lower() and 'error' not in description.lower():
                successful += 1
                print(f"  ✓ {description[:80]}...")
            else:
                print(f"  ❌ {description}")
            
            # Progress stats
            if (idx + 1) % 100 == 0:
                print(f"\n📊 Progress: {idx + 1}/{len(videos_to_process)} | Success: {successful}/{idx + 1}")
            
            # Rate limiting
            time.sleep(2)
        
        # Add descriptions to dataframe
        df_processed = videos_to_process.copy()
        df_processed['vision_description'] = descriptions
        
        print(f"\n✓ Processed {len(descriptions):,} videos")
        print(f"✓ Successful: {successful}/{len(descriptions)}")
        
        return df_processed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate descriptions using vision AI on videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process top 100 videos with LLaVA (free)
  python video_vision_descriptions.py --csv data.csv --top 100 --output described.csv
  
  # Process top 1000 with GPT-4 Vision (paid)
  python video_vision_descriptions.py --csv data.csv --top 1000 --vision gpt4v --output described.csv
  
  # Keep downloaded videos for later use
  python video_vision_descriptions.py --csv data.csv --top 100 --keep-videos

Requirements:
  pip install opencv-python yt-dlp --break-system-packages
  
  For LLaVA (free):
    ollama pull llava
  
  For GPT-4 Vision (paid):
    export OPENAI_API_KEY=sk-...
        """
    )
    
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--top', type=int, help='Process top N videos by engagement')
    parser.add_argument('--skip', type=int, default=0, help='Skip first N videos (for batching)')
    parser.add_argument('--vision', choices=['llava', 'gpt4v'], default='llava',
                       help='Vision model to use')
    parser.add_argument('--keep-videos', action='store_true',
                       help='Keep downloaded videos (uses storage)')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import cv2
    except ImportError:
        print("❌ opencv-python not installed!")
        print("Install: pip install opencv-python --break-system-packages")
        sys.exit(1)
    
    # Load data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded: {len(df):,} videos")
    
    # Sort by engagement if --top specified
    if args.top:
        print(f"\nSorting by engagement and taking top {args.top:,} (skipping first {args.skip:,})...")
        df['engagement'] = df.get('play_count', 0) + df.get('digg_count', 0) * 10
        df_sorted = df.sort_values('engagement', ascending=False)
        
        # Apply skip and top
        start_idx = args.skip
        end_idx = args.skip + args.top
        df = df_sorted.iloc[start_idx:end_idx]
        print(f"Selected videos {start_idx} to {end_idx} (total: {len(df):,})")
    
    # Process videos
    describer = VideoVisionDescriber(
        vision_provider=args.vision,
        keep_videos=args.keep_videos
    )
    
    df_described = describer.process_dataset(df, max_videos=args.top)
    
    # Save
    print(f"\nSaving to {args.output}...")
    df_described.to_csv(args.output, index=False)
    print(f"✓ Saved {len(df_described):,} videos with vision descriptions")
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE DESCRIPTIONS")
    print("="*80)
    for idx in range(min(3, len(df_described))):
        row = df_described.iloc[idx]
        print(f"\n[{idx + 1}]")
        print(f"Video ID: {row['id']}")
        print(f"Description: {row['vision_description']}")
        print("-"*80)


if __name__ == "__main__":
    main()