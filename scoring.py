import torch


def maxsim(query_embs, doc_embs, doc_mask):
    sim = torch.bmm(query_embs, doc_embs.transpose(1, 2))
    sim = sim.masked_fill(~doc_mask.unsqueeze(1).bool(), float('-inf'))
    return sim.max(dim=2).values.sum(dim=1)
