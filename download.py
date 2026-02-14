from datasets import load_dataset

# Load only the first 500,000 samples
dataset = load_dataset(
    "The-data-company/TikTok-10M",
    split="train[:500000]"
)

# Convert to pandas DataFrame and save as CSV
df = dataset.to_pandas()
df.to_csv("tiktok_500k.csv", index=False)

print(f"Saved {len(df)} samples to tiktok_500k.csv")
print(f"CSV size: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())