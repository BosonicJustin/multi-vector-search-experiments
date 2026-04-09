import argparse
import json
import os
import numpy as np
import bm25s
from datasets import load_dataset
from tqdm import tqdm


def collect_passages_and_triplets(split):
    raw = load_dataset("microsoft/ms_marco", "v1.1", split=split)

    all_passages = {}
    queries_with_positives = []

    for row in tqdm(raw, desc=f"Scanning {split}"):
        query = row["query"]
        passages = row["passages"]

        pos_indices = [i for i, val in enumerate(passages["is_selected"]) if val > 0]
        if not pos_indices:
            continue

        pos_texts = [passages["passage_text"][i] for i in pos_indices]

        for text in passages["passage_text"]:
            if text not in all_passages:
                all_passages[text] = len(all_passages)

        queries_with_positives.append({
            "query": query,
            "positives": pos_texts,
            "positive_ids": [all_passages[t] for t in pos_texts],
        })

    return all_passages, queries_with_positives


def build_bm25_index(passages):
    print(f"Building BM25 index over {len(passages)} passages...")
    texts = [""] * len(passages)
    for text, idx in passages.items():
        texts[idx] = text

    tokenized = bm25s.tokenize(texts, show_progress=True)
    index = bm25s.BM25()
    index.index(tokenized, show_progress=True)
    return texts, index


def mine_hard_negatives(queries, bm25_index, passage_texts, top_k=10):
    triplets = []

    query_texts = [item["query"] for item in queries]
    tokenized_queries = bm25s.tokenize(query_texts, show_progress=True)

    print("Retrieving top-k for all queries...")
    results, scores = bm25_index.retrieve(tokenized_queries, k=top_k, show_progress=True)

    for i, item in enumerate(tqdm(queries, desc="Building triplets")):
        positive_ids = set(item["positive_ids"])
        top_indices = results[i]

        hard_neg_idx = None
        for idx in top_indices:
            if idx not in positive_ids:
                hard_neg_idx = idx
                break

        if hard_neg_idx is None:
            continue

        hard_neg_text = passage_texts[hard_neg_idx]

        for pos_text in item["positives"]:
            triplets.append({
                "query": item["query"],
                "positive": pos_text,
                "negative": hard_neg_text,
            })

    return triplets


def preprocess_split(split, output_dir):
    all_passages, queries = collect_passages_and_triplets(split)
    print(f"{split}: {len(queries)} queries with positives, {len(all_passages)} unique passages")

    passage_texts, bm25_index = build_bm25_index(all_passages)
    triplets = mine_hard_negatives(queries, bm25_index, passage_texts)
    print(f"{split}: {len(triplets)} triplets")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split}.jsonl")
    with open(path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")
    print(f"Saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    for split in args.splits:
        preprocess_split(split, args.output_dir)
