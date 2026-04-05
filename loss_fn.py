import torch
from torch import nn
import torch.nn.functional as F

from scoring import maxsim


class MultivectorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_embs, pos_embs, pos_mask, neg_embs, neg_mask):
        batch_size, max_num_neg, max_doc_len, dim = neg_embs.shape

        pos_scores = maxsim(query_embs, pos_embs, pos_mask)

        q_exp = query_embs.unsqueeze(1).expand(-1, max_num_neg, -1, -1).reshape(-1, query_embs.size(1), dim)
        neg_embs_flat = neg_embs.reshape(-1, max_doc_len, dim)
        neg_mask_flat = neg_mask.reshape(-1, max_doc_len)

        neg_scores = maxsim(q_exp, neg_embs_flat, neg_mask_flat).view(batch_size, max_num_neg)

        neg_doc_padding = neg_mask.sum(dim=2) == 0
        neg_scores = neg_scores.masked_fill(neg_doc_padding, float('-inf'))

        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embs.device)

        return F.cross_entropy(all_scores, labels)
