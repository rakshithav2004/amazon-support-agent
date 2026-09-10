import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/amazon_threads.csv"

os.makedirs("data/processed", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# Keep only conversations that contain AmazonHelp tweets.
amazon_ids = set(
    df.loc[
        df["author_id"].astype(str).str.lower() == "amazonhelp",
        "tweet_id"
    ]
)

print(f"AmazonHelp tweets found: {len(amazon_ids):,}")

# Find tweets that directly received a response from AmazonHelp.
customer_messages = df[
    df["response_tweet_id"].notna()
    & df["response_tweet_id"].apply(
        lambda x: any(
            int(tweet_id) in amazon_ids
            for tweet_id in str(x).split(",")
            if tweet_id.strip().isdigit()
        )
    )
].copy()

# Keep useful columns
customer_messages = customer_messages[
    [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
]

# Add a column identifying the brand response
def find_amazon_reply_ids(value):
    if pd.isna(value):
        return ""

    ids = []

    for tweet_id in str(value).split(","):
        tweet_id = tweet_id.strip()

        if tweet_id.isdigit() and int(tweet_id) in amazon_ids:
            ids.append(tweet_id)

    return ",".join(ids)


customer_messages["amazon_response_tweet_id"] = (
    customer_messages["response_tweet_id"]
    .apply(find_amazon_reply_ids)
)

customer_messages.to_csv(OUTPUT_FILE, index=False)

print("\nDone!")
print(f"Customer messages with AmazonHelp responses: {len(customer_messages):,}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nExample:")
print(
    customer_messages[
        ["tweet_id", "author_id", "text", "amazon_response_tweet_id"]
    ].head(10).to_string(index=False)
)