
import pandas as pd
import glob

# Find all batch files
batch_files = sorted(glob.glob('batch*.csv'))
print(f'Found {len(batch_files)} batch files: {batch_files}')

# Load and merge all batches
batches = []
for file in batch_files:
    df = pd.read_csv(file)
    print(f'Loaded {file}: {len(df)} videos')
    batches.append(df)

# Combine
combined = pd.concat(batches, ignore_index=True)
print(f'\nTotal videos: {len(combined)}')

# Save merged file
combined.to_csv('tiktok_all_vision_described.csv', index=False)
print('✓ Saved to: tiktok_all_vision_described.csv')