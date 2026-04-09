import torch
from torch import nn
import torch.nn.functional as F

from scoring import maxsim


class MultivectorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_embs, pos_embs, pos_mask, neg_embs, neg_mask):
        pos_scores = maxsim(query_embs, pos_embs, pos_mask)
        neg_scores = maxsim(query_embs, neg_embs, neg_mask)

        all_scores = torch.stack([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(all_scores.size(0), dtype=torch.long, device=query_embs.device)

        return F.cross_entropy(all_scores, labels)
