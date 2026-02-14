"""
Advanced Niche Filtering Tool
Supports multiple filtering criteria
"""

import pandas as pd
import numpy as np

# ============ CONFIGURATION ============
# Set these to None to skip that filter, or a number to apply it

# Filter by minimum hashtags per niche
MIN_HASHTAGS = 50  # Remove niches with fewer hashtags

# Filter by percentile (alternative to MIN_HASHTAGS)
# Keep only niches in top X percentile by size
KEEP_TOP_PERCENTILE = None  # e.g., 75 = keep top 25% of niches by size

# Filter by total frequency (sum of all hashtag frequencies in niche)
MIN_TOTAL_FREQUENCY = None  # e.g., 1000 = niche hashtags must appear 1000+ times total

# Remove specific niches by name (case-insensitive)
REMOVE_NICHES = []  # e.g., ['fyp', 'viral', 'foryou']

# Keep only specific niches by name (leave empty to keep all)
KEEP_ONLY_NICHES = []  # e.g., ['Beauty & Makeup', 'Gaming']

# Files
SUMMARY_FILE = 'cluster_summary.csv'
MAPPING_FILE = 'hashtag_to_niche.csv'
# =======================================

print("="*60)
print("ADVANCED NICHE FILTERING")
print("="*60)

# Load data
print(f"\nLoading data...")
summary = pd.read_csv(SUMMARY_FILE)
mapping = pd.read_csv(MAPPING_FILE)

# Calculate total frequency per niche if needed
if MIN_TOTAL_FREQUENCY is not None:
    print("Calculating total frequencies...")
    niche_freq = mapping.groupby('cluster')['frequency'].sum().reset_index()
    niche_freq.columns = ['cluster_id', 'total_frequency']
    summary = summary.merge(niche_freq, on='cluster_id', how='left')

print(f"\nInitial state:")
print(f"  Niches: {len(summary)}")
print(f"  Hashtags: {len(mapping)}")

# Track what we're keeping
keep_mask = pd.Series([True] * len(summary), index=summary.index)
filter_reasons = []

# Apply filters
print(f"\nApplying filters:")

# Filter 1: Minimum hashtags
if MIN_HASHTAGS is not None:
    before = keep_mask.sum()
    keep_mask &= summary['num_hashtags'] >= MIN_HASHTAGS
    removed = before - keep_mask.sum()
    if removed > 0:
        print(f"  ✓ Min hashtags ({MIN_HASHTAGS}): removed {removed} niches")
        filter_reasons.append(f"min_hashtags_{MIN_HASHTAGS}")

# Filter 2: Percentile
if KEEP_TOP_PERCENTILE is not None:
    before = keep_mask.sum()
    threshold = np.percentile(summary['num_hashtags'], 100 - KEEP_TOP_PERCENTILE)
    keep_mask &= summary['num_hashtags'] >= threshold
    removed = before - keep_mask.sum()
    if removed > 0:
        print(f"  ✓ Top {KEEP_TOP_PERCENTILE}% by size: removed {removed} niches (threshold: {threshold:.0f})")
        filter_reasons.append(f"top_{KEEP_TOP_PERCENTILE}_percentile")

# Filter 3: Total frequency
if MIN_TOTAL_FREQUENCY is not None:
    before = keep_mask.sum()
    keep_mask &= summary['total_frequency'] >= MIN_TOTAL_FREQUENCY
    removed = before - keep_mask.sum()
    if removed > 0:
        print(f"  ✓ Min total frequency ({MIN_TOTAL_FREQUENCY}): removed {removed} niches")
        filter_reasons.append(f"min_freq_{MIN_TOTAL_FREQUENCY}")

# Filter 4: Remove specific niches
if REMOVE_NICHES:
    before = keep_mask.sum()
    remove_lower = [n.lower() for n in REMOVE_NICHES]
    keep_mask &= ~summary['niche_name'].str.lower().isin(remove_lower)
    removed = before - keep_mask.sum()
    if removed > 0:
        print(f"  ✓ Blacklist: removed {removed} niches ({', '.join(REMOVE_NICHES)})")
        filter_reasons.append("blacklist")

# Filter 5: Keep only specific niches
if KEEP_ONLY_NICHES:
    before = keep_mask.sum()
    keep_lower = [n.lower() for n in KEEP_ONLY_NICHES]
    keep_mask &= summary['niche_name'].str.lower().isin(keep_lower)
    removed = before - keep_mask.sum()
    if removed > 0:
        print(f"  ✓ Whitelist: kept only {keep_mask.sum()} niches")
        filter_reasons.append("whitelist")

# Get filtered data
filtered_summary = summary[keep_mask].copy()
removed_summary = summary[~keep_mask].copy()

# Filter mapping
keep_cluster_ids = set(filtered_summary['cluster_id'])
filtered_mapping = mapping[mapping['cluster'].isin(keep_cluster_ids)].copy()
removed_mapping = mapping[~mapping['cluster'].isin(keep_cluster_ids)].copy()

print(f"\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nKept:")
print(f"  Niches: {len(filtered_summary)}")
print(f"  Hashtags: {len(filtered_mapping)}")

print(f"\nRemoved:")
print(f"  Niches: {len(removed_summary)}")
print(f"  Hashtags: {len(removed_mapping)}")

# Show what was removed
if len(removed_summary) > 0:
    print(f"\n" + "="*60)
    print("REMOVED NICHES")
    print("="*60)
    display_cols = ['niche_name', 'num_hashtags']
    if 'total_frequency' in removed_summary.columns:
        display_cols.append('total_frequency')
    display_cols.append('top_5_hashtags')
    
    print(removed_summary[display_cols].to_string(index=False, max_rows=20))

# Show top kept niches
if len(filtered_summary) > 0:
    print(f"\n" + "="*60)
    print("TOP 10 KEPT NICHES")
    print("="*60)
    display_cols = ['niche_name', 'num_hashtags']
    if 'total_frequency' in filtered_summary.columns:
        display_cols.append('total_frequency')
    display_cols.append('top_5_hashtags')
    
    print(filtered_summary.nlargest(10, 'num_hashtags')[display_cols].to_string(index=False))

# Save results
if len(filtered_summary) > 0:
    # Create descriptive filename
    filter_suffix = '_'.join(filter_reasons) if filter_reasons else 'filtered'
    output_summary = SUMMARY_FILE.replace('.csv', f'_{filter_suffix}.csv')
    output_mapping = MAPPING_FILE.replace('.csv', f'_{filter_suffix}.csv')
    
    filtered_summary.to_csv(output_summary, index=False)
    filtered_mapping.to_csv(output_mapping, index=False)
    
    print(f"\n" + "="*60)
    print("SAVED FILES")
    print("="*60)
    print(f"\n✓ {output_summary}")
    print(f"✓ {output_mapping}")
    
    # Optional: save removed items too
    if len(removed_mapping) > 0:
        removed_output = MAPPING_FILE.replace('.csv', '_removed.csv')
        removed_mapping.to_csv(removed_output, index=False)
        print(f"✓ {removed_output} (removed hashtags for reference)")
    
    print(f"\n✓ Done! Filtered dataset ready to use.")
else:
    print("\n⚠ Warning: No niches remaining after filtering!")
    print("Try adjusting your filter criteria.")