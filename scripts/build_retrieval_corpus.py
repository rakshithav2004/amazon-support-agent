import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
THREAD_FILE = "data/processed/amazon_threads.csv"
OUTPUT_FILE = "data/processed/amazon_retrieval_corpus.csv"


print("Loading original dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Total tweets loaded: {len(df):,}")


print("\nLoading Amazon customer-response pairs...")

threads = pd.read_csv(
    THREAD_FILE,
    low_memory=False
)

print(
    f"Customer messages with AmazonHelp responses: "
    f"{len(threads):,}"
)


# --------------------------------------------------
# Create tweet lookup
# --------------------------------------------------

tweet_lookup = (
    df[
        [
            "tweet_id",
            "text",
            "author_id",
            "inbound",
            "created_at"
        ]
    ]
    .drop_duplicates("tweet_id")
    .set_index("tweet_id")
)


# --------------------------------------------------
# Build retrieval records
# --------------------------------------------------

records = []


for _, row in threads.iterrows():

    customer_tweet_id = int(row["tweet_id"])

    response_ids = str(
        row["amazon_response_tweet_id"]
    ).split(",")

    customer_text = row["text"]

    for response_id in response_ids:

        response_id = response_id.strip()

        if not response_id.isdigit():
            continue

        response_id = int(response_id)

        if response_id not in tweet_lookup.index:
            continue

        response = tweet_lookup.loc[response_id]

        response_text = response["text"]

        if pd.isna(response_text):
            continue

        if not str(customer_text).strip():
            continue

        if not str(response_text).strip():
            continue

        records.append(
            {
                "evidence_id": f"{customer_tweet_id}_{response_id}",
                "customer_tweet_id": customer_tweet_id,
                "reply_tweet_id": response_id,
                "customer_text": str(customer_text),
                "reply_text": str(response_text),
                "reply_created_at": response["created_at"]
            }
        )


# --------------------------------------------------
# Save corpus
# --------------------------------------------------

corpus = pd.DataFrame(records)


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


corpus.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================")
print("RETRIEVAL CORPUS")
print("==============================")

print(
    f"Historical examples: {len(corpus):,}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


print("\nExample records:")

print(
    corpus.head(10).to_string(
        index=False
    )
)