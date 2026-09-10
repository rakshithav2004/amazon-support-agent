import os

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

CORPUS_FILE = "data/processed/amazon_retrieval_corpus_clean.csv"
INDEX_FILE = "data/processed/amazon_retrieval.index"
METADATA_FILE = "data/processed/amazon_retrieval_metadata.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


class HistoricalRetriever:

    def __init__(self):

        print("Loading retrieval corpus...")

        self.corpus = pd.read_csv(
            CORPUS_FILE,
            low_memory=False
        )

        print(
            f"Historical examples: "
            f"{len(self.corpus):,}"
        )

        print(
            f"Loading embedding model: "
            f"{MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        self.index = None

        if os.path.exists(INDEX_FILE):

            print("Loading existing FAISS index...")

            self.index = faiss.read_index(
                INDEX_FILE
            )

        else:

            self.build_index()


    def build_index(self):

        print("\nCreating embeddings...")

        texts = (
            self.corpus["customer_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        print(
            f"Embedding dimension: {dimension}"
        )

        print("Building FAISS index...")

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        os.makedirs(
            os.path.dirname(INDEX_FILE),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        self.corpus.to_csv(
            METADATA_FILE,
            index=False
        )

        print(
            f"\nFAISS index saved to: "
            f"{INDEX_FILE}"
        )

        print(
            f"Metadata saved to: "
            f"{METADATA_FILE}"
        )


    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            row = self.corpus.iloc[int(index)]

            results.append(
                {
                    "evidence_id": row["evidence_id"],
                    "customer_tweet_id": row[
                        "customer_tweet_id"
                    ],
                    "reply_tweet_id": row[
                        "reply_tweet_id"
                    ],
                    "customer_text": row[
                        "customer_text"
                    ],
                    "reply_text": row[
                        "reply_text"
                    ],
                    "score": float(score)
                }
            )

        return results


if __name__ == "__main__":

    retriever = HistoricalRetriever()

    query = (
        "My package says delivered but "
        "I never received it."
    )

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n==============================")
    print("RETRIEVAL RESULTS")
    print("==============================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Evidence ID: "
            f"{result['evidence_id']}"
        )

        print(
            f"Customer: "
            f"{result['customer_text']}"
        )

        print(
            f"AmazonHelp: "
            f"{result['reply_text']}"
        )