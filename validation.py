import torch
import torch.nn.functional as F
import numpy as np

from scoring import maxsim


K_VALUES = [5, 50, 100]
ENCODE_BATCH_SIZE = 64
SCORE_BATCH_SIZE = 64


def dcg_at_k(relevances, k):
    rel = np.array(relevances[:k], dtype=np.float64)
    positions = np.arange(1, len(rel) + 1)
    return np.sum(rel / np.log2(positions + 1))


def ndcg_at_k(relevances, k):
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def encode_texts(encoder, texts, batch_size):
    chunks = []
    mask_chunks = []
    max_len = 0

    for i in range(0, len(texts), batch_size):
        embs, mask = encoder(texts[i:i + batch_size])
        chunks.append(embs)
        mask_chunks.append(mask)
        max_len = max(max_len, embs.size(1))

    for i in range(len(chunks)):
        pad = max_len - chunks[i].size(1)
        if pad > 0:
            chunks[i] = F.pad(chunks[i], (0, 0, 0, pad))
            mask_chunks[i] = F.pad(mask_chunks[i], (0, pad))

    return torch.cat(chunks), torch.cat(mask_chunks)


def validate(query_encoder, doc_encoder, val_loader, device, max_batches=None):
    query_encoder.eval()
    doc_encoder.eval()

    all_queries = []
    doc_pool = []
    doc_text_to_idx = {}
    query_relevant_docs = []

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break
        for i in range(len(batch["queries"])):
            query = batch["queries"][i]
            positive = batch["positives"][i]
            negatives = batch["negative_lists"][i]

            all_queries.append(query)

            if positive not in doc_text_to_idx:
                doc_text_to_idx[positive] = len(doc_pool)
                doc_pool.append(positive)

            for neg in negatives:
                if neg not in doc_text_to_idx:
                    doc_text_to_idx[neg] = len(doc_pool)
                    doc_pool.append(neg)

            query_relevant_docs.append({doc_text_to_idx[positive]})

    n_queries = len(all_queries)
    n_docs = len(doc_pool)
    print(f"Validation: {n_queries} queries, {n_docs} unique documents")

    with torch.no_grad():
        query_embs, _ = encode_texts(query_encoder, all_queries, ENCODE_BATCH_SIZE)
        doc_embs, doc_masks = encode_texts(doc_encoder, doc_pool, ENCODE_BATCH_SIZE)

        all_scores = torch.zeros(n_queries, n_docs, device=device)

        for q_start in range(0, n_queries, SCORE_BATCH_SIZE):
            q_end = min(q_start + SCORE_BATCH_SIZE, n_queries)
            q_batch = query_embs[q_start:q_end]
            q_size = q_end - q_start

            for d_start in range(0, n_docs, SCORE_BATCH_SIZE):
                d_end = min(d_start + SCORE_BATCH_SIZE, n_docs)
                d_batch = doc_embs[d_start:d_end]
                dm_batch = doc_masks[d_start:d_end]
                d_size = d_end - d_start

                q_exp = q_batch.unsqueeze(1).expand(-1, d_size, -1, -1).reshape(-1, q_batch.size(1), q_batch.size(2))
                d_exp = d_batch.unsqueeze(0).expand(q_size, -1, -1, -1).reshape(-1, d_batch.size(1), d_batch.size(2))
                dm_exp = dm_batch.unsqueeze(0).expand(q_size, -1, -1).reshape(-1, dm_batch.size(1))

                scores = maxsim(q_exp, d_exp, dm_exp).view(q_size, d_size)
                all_scores[q_start:q_end, d_start:d_end] = scores

    labels = torch.tensor([list(rel)[0] for rel in query_relevant_docs], device=device)
    val_loss = F.cross_entropy(all_scores, labels).item()

    all_metrics = {f"{m}@{k}": [] for k in K_VALUES for m in ["recall", "precision", "ndcg"]}
    all_metrics["loss"] = val_loss
    all_scores_np = all_scores.cpu().numpy()

    for i in range(n_queries):
        relevant = query_relevant_docs[i]
        ranked_indices = np.argsort(-all_scores_np[i])
        relevances = [1 if idx in relevant else 0 for idx in ranked_indices]

        for k in K_VALUES:
            top_k = set(ranked_indices[:k].tolist())
            hits = len(top_k & relevant)
            all_metrics[f"recall@{k}"].append(hits / len(relevant))
            all_metrics[f"precision@{k}"].append(hits / k)
            all_metrics[f"ndcg@{k}"].append(ndcg_at_k(relevances, k))

    return {key: np.mean(vals) for key, vals in all_metrics.items() if vals}
