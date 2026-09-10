import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/amazon_help.csv"

os.makedirs("data/processed", exist_ok=True)

chunks = []
total_rows = 0

print("Reading dataset...")

for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):
    total_rows += len(chunk)

    # Amazon's support account
    amazon = chunk[
        chunk["author_id"].astype(str).str.lower() == "amazonhelp"
    ]

    if not amazon.empty:
        chunks.append(amazon)

    if total_rows % 500000 == 0:
        print(f"Processed {total_rows:,} rows...")

if chunks:
    amazon_df = pd.concat(chunks, ignore_index=True)
    amazon_df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone!")
    print(f"AmazonHelp tweets: {len(amazon_df):,}")
    print(f"Saved to: {OUTPUT_FILE}")
else:
    print("\nNo AmazonHelp tweets found.")