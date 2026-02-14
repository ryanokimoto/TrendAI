#!/usr/bin/env python3
"""
Smart Metadata Descriptions (No Video Download Required)
Combines existing metadata fields to create rich descriptions
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, Any
import time


class MetadataDescriptionGenerator:
    """Generate descriptions from TikTok metadata without downloading videos"""
    
    def __init__(self, use_llm_enhancement: bool = False):
        """
        Args:
            use_llm_enhancement: Use Ollama to make descriptions more natural
        """
        self.use_llm_enhancement = use_llm_enhancement
        
        if use_llm_enhancement:
            self.setup_llm()
    
    def setup_llm(self):
        """Setup Ollama for description enhancement"""
        import subprocess
        
        try:
            # Test if Ollama is available
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'llama' not in result.stdout.lower():
                print("⚠️  Ollama is installed but no model found")
                print("Run: ollama pull llama3.2")
                self.use_llm_enhancement = False
            else:
                print("✓ Ollama ready for enhancement")
                
        except Exception:
            print("⚠️  Ollama not available, skipping LLM enhancement")
            print("Install from https://ollama.ai for better descriptions")
            self.use_llm_enhancement = False
    
    def parse_hashtags(self, challenges_field: Any) -> list:
        """Parse hashtags from challenges field"""
        hashtags = []
        
        if pd.isna(challenges_field):
            return hashtags
        
        challenges_str = str(challenges_field)
        
        # Try parsing as JSON
        try:
            challenges_list = json.loads(challenges_str)
            if isinstance(challenges_list, list):
                for item in challenges_list:
                    if isinstance(item, dict) and 'title' in item:
                        hashtags.append(item['title'])
                    elif isinstance(item, str):
                        hashtags.append(item)
        except:
            # Try as comma-separated
            if ',' in challenges_str:
                hashtags = [h.strip() for h in challenges_str.split(',')]
            else:
                hashtags = [challenges_str]
        
        return [h for h in hashtags if h][:10]  # Limit to 10
    
    def categorize_by_hashtags(self, hashtags: list) -> str:
        """Infer category from hashtags"""
        hashtags_lower = [h.lower() for h in hashtags]
        
        # Category mapping
        categories = {
            'cooking': ['cooking', 'recipe', 'food', 'chef', 'baking', 'kitchen'],
            'fitness': ['fitness', 'workout', 'gym', 'exercise', 'health', 'yoga'],
            'comedy': ['funny', 'comedy', 'humor', 'prank', 'laugh', 'meme'],
            'education': ['education', 'learning', 'tutorial', 'howto', 'tips', 'guide'],
            'beauty': ['beauty', 'makeup', 'skincare', 'hair', 'cosmetics'],
            'fashion': ['fashion', 'style', 'outfit', 'ootd', 'clothing'],
            'travel': ['travel', 'vacation', 'trip', 'explore', 'wanderlust'],
            'music': ['music', 'singing', 'dance', 'song', 'cover'],
            'gaming': ['gaming', 'game', 'gamer', 'gameplay', 'esports'],
            'pets': ['pets', 'dog', 'cat', 'animal', 'puppy', 'kitten'],
            'diy': ['diy', 'craft', 'handmade', 'creative', 'art'],
            'tech': ['tech', 'technology', 'gadget', 'software', 'coding'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in ' '.join(hashtags_lower) for keyword in keywords):
                return category
        
        return 'general'
    
    def create_description(self, row: pd.Series) -> str:
        """
        Create rich description from metadata fields
        
        Uses:
        - desc: Author caption
        - challenges: Hashtags
        - music_title, music_author_name: Audio context  
        - poi_name, city: Location
        - play_count, digg_count: Engagement
        """
        parts = []
        
        # 1. Author description (if meaningful)
        desc = str(row.get('desc', '')).strip()
        
        # Filter out low-quality captions
        skip_patterns = ['fyp', 'viral', 'wait for it', 'watch till end', '😂', '🔥', '💯']
        is_meaningful = (
            desc and 
            len(desc) > 15 and 
            not all(c in '😂🔥💯#@' for c in desc) and
            not any(pattern in desc.lower() for pattern in skip_patterns)
        )
        
        if is_meaningful:
            parts.append(desc[:200])  # Limit length
        
        # 2. Hashtag-based category
        hashtags = self.parse_hashtags(row.get('challenges'))
        category = self.categorize_by_hashtags(hashtags)
        
        if category != 'general':
            parts.append(f"A {category} video")
        
        # Add top hashtags
        if hashtags:
            hashtag_text = ', '.join([f"#{h}" for h in hashtags[:5]])
            parts.append(f"Topics: {hashtag_text}")
        
        # 3. Music context
        music_title = str(row.get('music_title', '')).strip()
        music_author = str(row.get('music_author_name', '')).strip()
        
        if music_title and music_title.lower() not in ['nan', 'none', 'original sound']:
            if music_author and music_author.lower() not in ['nan', 'none']:
                parts.append(f"Music: '{music_title}' by {music_author}")
            else:
                parts.append(f"Music: '{music_title}'")
        elif music_title and 'original' in music_title.lower():
            parts.append("Original audio")
        
        # 4. Location context
        poi_name = str(row.get('poi_name', '')).strip()
        city = str(row.get('city', '')).strip()
        
        if poi_name and poi_name.lower() not in ['nan', 'none']:
            if city and city.lower() not in ['nan', 'none']:
                parts.append(f"Filmed at {poi_name}, {city}")
            else:
                parts.append(f"Filmed at {poi_name}")
        elif city and city.lower() not in ['nan', 'none']:
            parts.append(f"Location: {city}")
        
        # 5. Engagement signals
        views = row.get('play_count', 0)
        likes = row.get('digg_count', 0)
        
        if views > 10000000:  # 10M+
            parts.append("Viral video")
        elif views > 1000000:  # 1M+
            parts.append("Popular video")
        
        # Combine all parts
        description = '. '.join(parts)
        
        # Fallback if nothing meaningful
        if not description or len(description) < 20:
            description = f"A {category} TikTok video"
            if hashtags:
                description += f" about {', '.join(hashtags[:3])}"
        
        return description
    
    def enhance_with_llm(self, metadata_desc: str) -> str:
        """Use Ollama to make description more natural"""
        if not self.use_llm_enhancement:
            return metadata_desc
        
        import subprocess
        
        prompt = f"""Rewrite this TikTok metadata as a natural 2-3 sentence description. Keep it factual and concise.

Metadata: {metadata_desc}

Natural description (2-3 sentences):"""
        
        try:
            result = subprocess.run(
                ['ollama', 'run', 'llama3.2', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            enhanced = result.stdout.strip()
            
            # Use enhanced if it's good, otherwise fallback
            if enhanced and len(enhanced) > 20 and len(enhanced) < 500:
                return enhanced
            else:
                return metadata_desc
                
        except Exception:
            return metadata_desc
    
    def process_dataset(self, 
                       df: pd.DataFrame,
                       batch_size: int = 1000,
                       show_progress: bool = True) -> pd.DataFrame:
        """Process entire dataset and add descriptions"""
        
        print("\n" + "="*80)
        print("GENERATING METADATA-BASED DESCRIPTIONS")
        print("="*80)
        print(f"Videos: {len(df):,}")
        print(f"LLM Enhancement: {'Yes (Ollama)' if self.use_llm_enhancement else 'No'}")
        print("="*80 + "\n")
        
        descriptions = []
        
        for idx, row in df.iterrows():
            # Create base description
            desc = self.create_description(row)
            
            # Optional LLM enhancement
            if self.use_llm_enhancement:
                desc = self.enhance_with_llm(desc)
                time.sleep(0.5)  # Rate limiting
            
            descriptions.append(desc)
            
            # Progress
            if show_progress and (idx + 1) % batch_size == 0:
                print(f"Processed {idx + 1:,}/{len(df):,} videos...")
        
        df['ai_description'] = descriptions
        
        print(f"\n✓ Generated {len(descriptions):,} descriptions")
        
        return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate descriptions from metadata (no video download)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Just metadata combinations
  python metadata_descriptions.py --csv data.csv --output described.csv
  
  # Enhanced: Use Ollama to make it more natural
  python metadata_descriptions.py --csv data.csv --enhance --output described.csv
  
  # Test with sample first
  python metadata_descriptions.py --csv data.csv --sample 1000 --output test.csv
        """
    )
    
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--sample', type=int, help='Process only N rows (for testing)')
    parser.add_argument('--enhance', action='store_true', 
                       help='Use Ollama to enhance descriptions')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.csv}...")
    if args.sample:
        df = pd.read_csv(args.csv, nrows=args.sample)
        print(f"Loaded sample: {len(df):,} videos")
    else:
        df = pd.read_csv(args.csv)
        print(f"Loaded: {len(df):,} videos")
    
    # Generate descriptions
    generator = MetadataDescriptionGenerator(use_llm_enhancement=args.enhance)
    df_described = generator.process_dataset(df)
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE DESCRIPTIONS")
    print("="*80)
    for idx in range(min(3, len(df_described))):
        row = df_described.iloc[idx]
        print(f"\n[{idx + 1}]")
        print(f"Original caption: {row.get('desc', 'N/A')[:80]}...")
        print(f"Generated description: {row['ai_description']}")
        print("-"*80)
    
    # Save
    print(f"\nSaving to {args.output}...")
    df_described.to_csv(args.output, index=False)
    print(f"✓ Saved {len(df_described):,} videos with descriptions")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Build search index:")
    print(f"   python convert_existing_dataset.py --csv {args.output} --full")
    print("\n2. Or enhance top videos with vision AI:")
    print(f"   python video_vision_descriptions.py --csv {args.output} --top 10000")
    print("="*80)


if __name__ == "__main__":
    main()