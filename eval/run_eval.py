import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from colbert_encoders import ColbertEncoder
from scoring import maxsim

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "..", "embeddings")
ENCODE_BATCH_SIZE = 64
INDEX_BATCH = 512
RECALL_KS = [50, 200, 1000]


def load_queries(path):
    queries = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            queries[row[0]] = row[1]
    return queries


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, _, pid, label = row
            if int(label) > 0:
                qrels[qid].add(pid)
    return qrels


def load_top1000(path, qids=None):
    top1000 = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            qid, pid, query_text, passage_text = row
            if qids and qid not in qids:
                continue
            top1000[qid].append({"pid": pid, "passage": passage_text})
    return top1000


def load_collection(path):
    passages = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            passages[row[0]] = row[1]
    return passages


def encode_texts(encoder, encode_fn, texts, batch_size):
    chunks = []
    mask_chunks = []
    max_len = 0
    for i in range(0, len(texts), batch_size):
        embs, mask = encode_fn(texts[i:i + batch_size])
        chunks.append(embs)
        mask_chunks.append(mask)
        max_len = max(max_len, embs.size(1))
    for i in range(len(chunks)):
        pad = max_len - chunks[i].size(1)
        if pad > 0:
            chunks[i] = F.pad(chunks[i], (0, 0, 0, pad))
            mask_chunks[i] = F.pad(mask_chunks[i], (0, pad))
    return torch.cat(chunks), torch.cat(mask_chunks)


def mrr_at_k(ranked_pids, relevant_pids, k=10):
    for i, pid in enumerate(ranked_pids[:k]):
        if pid in relevant_pids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_pids, relevant_pids, k):
    top_k = set(ranked_pids[:k])
    return len(top_k & relevant_pids) / len(relevant_pids)


def dcg_at_k(ranked_pids, relevant_pids, k):
    dcg = 0.0
    for i, pid in enumerate(ranked_pids[:k]):
        rel = 1.0 if pid in relevant_pids else 0.0
        dcg += rel / np.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_pids, relevant_pids, k):
    actual = dcg_at_k(ranked_pids, relevant_pids, k)
    ideal_pids = list(relevant_pids) + [p for p in ranked_pids if p not in relevant_pids]
    ideal = dcg_at_k(ideal_pids, relevant_pids, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def compute_all_metrics(ranked_pids, relevant_pids):
    m = {"MRR@10": mrr_at_k(ranked_pids, relevant_pids, 10)}
    for k in RECALL_KS:
        m[f"Recall@{k}"] = recall_at_k(ranked_pids, relevant_pids, k)
        m[f"NDCG@{k}"] = ndcg_at_k(ranked_pids, relevant_pids, k)
    return m


def aggregate(all_metrics):
    keys = all_metrics[0].keys()
    return {k: np.mean([m[k] for m in all_metrics]) for k in keys}


def eval_bm25(eval_qids, top1000, qrels):
    print("Evaluating BM25 baseline...")
    all_metrics = []
    for qid in tqdm(eval_qids):
        ranked_pids = [c["pid"] for c in top1000[qid]]
        all_metrics.append(compute_all_metrics(ranked_pids, qrels[qid]))
    return aggregate(all_metrics)


def eval_colbert_rerank(encoder, eval_qids, queries, top1000, qrels):
    print("Evaluating ColBERT re-ranking...")
    all_metrics = []
    with torch.no_grad():
        for qid in tqdm(eval_qids):
            query_text = queries[qid]
            candidates = top1000[qid]
            relevant_pids = qrels[qid]

            passage_texts = [c["passage"] for c in candidates]
            pids = [c["pid"] for c in candidates]

            query_embs, _ = encoder.encode_queries([query_text])
            doc_embs, doc_masks = encode_texts(encoder, encoder.encode_documents, passage_texts, ENCODE_BATCH_SIZE)

            q_exp = query_embs.expand(len(passage_texts), -1, -1)
            scores = maxsim(q_exp, doc_embs, doc_masks)

            ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()
            ranked_pids = [pids[i] for i in ranked_indices]

            all_metrics.append(compute_all_metrics(ranked_pids, relevant_pids))

    return aggregate(all_metrics)


def encode_collection_to_disk(encoder, collection, all_pids, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    embs_path = os.path.join(cache_dir, "token_embs.bin")
    meta_path = os.path.join(cache_dir, "meta.json")

    if os.path.exists(embs_path) and os.path.exists(meta_path):
        print(f"Loading cached embeddings from {cache_dir}")
        with open(meta_path) as f:
            meta = json.load(f)
        embs = np.memmap(embs_path, dtype=np.float16, mode="r",
                         shape=(meta["total_tokens"], meta["dim"]))
        return embs, meta["offsets"]

    print(f"Encoding {len(all_pids)} passages to {cache_dir}...")
    dim = encoder.search_dim
    offsets = []
    total_tokens = 0

    with open(embs_path, "wb") as f, torch.no_grad():
        for i in tqdm(range(0, len(all_pids), INDEX_BATCH), desc="Encoding"):
            batch_pids = all_pids[i:i + INDEX_BATCH]
            batch_texts = [collection[pid] for pid in batch_pids]

            batch_embs, batch_masks = encode_texts(encoder, encoder.encode_documents, batch_texts, ENCODE_BATCH_SIZE)
            batch_embs = batch_embs.cpu().numpy().astype(np.float16)
            batch_masks = batch_masks.cpu().numpy()

            for j, pid in enumerate(batch_pids):
                n = int(batch_masks[j].sum())
                valid = batch_embs[j, :n]
                f.write(valid.tobytes())
                offsets.append([total_tokens, n, pid])
                total_tokens += n

    print(f"Total tokens: {total_tokens}, dim: {dim}")
    print(f"Disk size: {total_tokens * dim * 2 / 1e9:.2f} GB")

    with open(meta_path, "w") as f:
        json.dump({"total_tokens": total_tokens, "dim": dim, "offsets": offsets}, f)

    embs = np.memmap(embs_path, dtype=np.float16, mode="r", shape=(total_tokens, dim))
    return embs, offsets


def select_passages(qrels, eval_qids, all_pids_in_collection, max_passages):
    """Always include passages relevant to eval queries; fill the rest randomly."""
    all_pids_set = set(all_pids_in_collection)
    must_include = set()
    for qid in eval_qids:
        for pid in qrels[qid]:
            if pid in all_pids_set:
                must_include.add(pid)

    if max_passages is None or max_passages >= len(all_pids_in_collection):
        return all_pids_in_collection, len(must_include)

    remaining = max_passages - len(must_include)
    if remaining <= 0:
        return list(must_include), len(must_include)

    candidates = [pid for pid in all_pids_in_collection if pid not in must_include]
    rng = np.random.default_rng(seed=42)
    sampled = rng.choice(len(candidates), size=remaining, replace=False)
    selected = list(must_include) + [candidates[i] for i in sampled]
    return selected, len(must_include)


def eval_colbert_e2e(encoder, eval_qids, queries, qrels, collection, max_passages, args):
    print("Loading collection passage IDs...")
    all_pids_in_collection = sorted(collection.keys(), key=int)
    selected_pids, n_relevant = select_passages(qrels, eval_qids, all_pids_in_collection, max_passages)
    print(f"Selected {len(selected_pids)} passages ({n_relevant} known relevant)")

    cache_dir = os.path.join(EMBED_DIR, f"step_{args.step}_n{len(selected_pids)}")
    embs_memmap, offsets = encode_collection_to_disk(encoder, collection, selected_pids, cache_dir)

    embs_f32 = np.array(embs_memmap, dtype=np.float32)
    print(f"Loaded embeddings into memory: {embs_f32.shape} ({embs_f32.nbytes / 1e9:.2f} GB)")

    import faiss
    print("Building FAISS index...")
    cpu_index = faiss.IndexFlatIP(encoder.search_dim)
    cpu_index.add(embs_f32)
    if faiss.get_num_gpus() > 0:
        print(f"Moving index to GPU ({faiss.get_num_gpus()} available)")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        print("Using CPU FAISS")
        index = cpu_index
    print(f"FAISS index size: {index.ntotal}")

    from bisect import bisect_right
    start_offsets = [t[0] for t in offsets]

    def token_idx_to_pid(token_idx):
        pos = bisect_right(start_offsets, token_idx) - 1
        return offsets[pos][2]

    print("Running end-to-end retrieval...")
    all_metrics = []
    
    
    
    
    
    
    
     torch.no_grad():
        for qid in tqdm(eval_qids):
            query_text = queries[qid]
            relevant_pids = qrels[qid]

            query_embs, _ = encoder.encode_queries([query_text])
            q_np = query_embs.squeeze(0).cpu().numpy().astype(np.float32)

            _, indices_per_token = index.search(q_np, args.faiss_topk)

            candidate_pids = set()
            for q_tok in range(q_np.shape[0]):
                for token_idx in indices_per_token[q_tok]:
                    candidate_pids.add(token_idx_to_pid(int(token_idx)))

            candidate_pids = list(candidate_pids)
            if not candidate_pids:
                all_metrics.append(compute_all_metrics([], relevant_pids))
                continue

            passage_texts = [collection[pid] for pid in candidate_pids]
            doc_embs, doc_masks = encode_texts(encoder, encoder.encode_documents, passage_texts, ENCODE_BATCH_SIZE)

            q_exp = query_embs.expand(len(candidate_pids), -1, -1)
            exact_scores = maxsim(q_exp, doc_embs, doc_masks)

            ranked_indices = torch.argsort(exact_scores, descending=True).cpu().tolist()
            ranked_pids = [candidate_pids[i] for i in ranked_indices]

            all_metrics.append(compute_all_metrics(ranked_pids, relevant_pids))

    return aggregate(all_metrics)


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{args.step}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("Loading data...")
    queries = load_queries(os.path.join(DATA_DIR, "queries.dev.tsv"))
    qrels = load_qrels(os.path.join(DATA_DIR, "qrels.dev.tsv"))

    eval_qids = [qid for qid in qrels if qid in queries]
    if args.max_queries:
        eval_qids = eval_qids[:args.max_queries]
        print(f"Limiting to {len(eval_qids)} queries")

    print(f"Loading checkpoint: {checkpoint_path}")
    encoder = ColbertEncoder()
    ckpt = torch.load(checkpoint_path, map_location=encoder.device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    output_path = os.path.join(RESULTS_DIR, f"eval_step_{args.step}.json")
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = {"step": args.step}

    if args.mode in ("rerank", "all"):
        top1000 = load_top1000(os.path.join(DATA_DIR, "top1000.dev.tsv"), qids=set(eval_qids))
        rerank_qids = [qid for qid in eval_qids if qid in top1000]
        print(f"Re-ranking on {len(rerank_qids)} queries")

        if "bm25" not in results:
            results["bm25"] = eval_bm25(rerank_qids, top1000, qrels)
            print(f"BM25: MRR@10={results['bm25']['MRR@10']:.4f}")

        results["colbert_rerank"] = eval_colbert_rerank(encoder, rerank_qids, queries, top1000, qrels)
        print(f"ColBERT Re-rank: MRR@10={results['colbert_rerank']['MRR@10']:.4f}")

    if args.mode in ("e2e", "all"):
        print("Loading passage collection...")
        collection = load_collection(os.path.join(DATA_DIR, "collection.tsv"))
        print(f"Total passages in collection: {len(collection)}")

        results["colbert_e2e"] = eval_colbert_e2e(
            encoder, eval_qids, queries, qrels, collection, args.max_passages, args
        )
        print(f"ColBERT E2E: MRR@10={results['colbert_e2e']['MRR@10']:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", choices=["rerank", "e2e", "all"], default="rerank")
    parser.add_argument("--max_passages", type=int, default=None,
                        help="Limit corpus size for e2e (None=use full 8.8M, requires huge memory)")
    parser.add_argument("--faiss_topk", type=int, default=100,
                        help="Top-k tokens to retrieve per query token in FAISS")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Limit number of dev queries to evaluate")
    main(parser.parse_args())
