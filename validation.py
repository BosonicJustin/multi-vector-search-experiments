import csv
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from scoring import maxsim


ENCODE_BATCH_SIZE = 64
SCORE_BATCH_SIZE = 64


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


RECALL_KS = [50, 200, 1000]


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


def validate(encoder, device, queries_path, qrels_path, top1000_path, max_queries=None):
    encoder.eval()

    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    eval_qids = [qid for qid in qrels if qid in queries]
    if max_queries:
        eval_qids = eval_qids[:max_queries]

    top1000 = load_top1000(top1000_path, qids=set(eval_qids))
    eval_qids = [qid for qid in eval_qids if qid in top1000]

    print(f"Validation: {len(eval_qids)} queries")

    all_mrr = []
    all_recall = {k: [] for k in RECALL_KS}
    all_ndcg = {k: [] for k in RECALL_KS}

    with torch.no_grad():
        for qid in eval_qids:
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

            all_mrr.append(mrr_at_k(ranked_pids, relevant_pids, k=10))
            for k in RECALL_KS:
                all_recall[k].append(recall_at_k(ranked_pids, relevant_pids, k))
                all_ndcg[k].append(ndcg_at_k(ranked_pids, relevant_pids, k))

    metrics = {"MRR@10": np.mean(all_mrr)}
    for k in RECALL_KS:
        metrics[f"Recall@{k}"] = np.mean(all_recall[k])
        metrics[f"NDCG@{k}"] = np.mean(all_ndcg[k])
    return metrics
