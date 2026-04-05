from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

corpus = load_dataset("microsoft/ms_marco", "v1.1")

def preprocess_expanded_negatives(batch):
    out_queries = []
    out_positives = []
    out_negatives = []

    for passages, query in zip(batch['passages'], batch['query']):
        pos_indices = [i for i, val in enumerate(passages['is_selected']) if val > 0]
        neg_indices = [i for i, val in enumerate(passages['is_selected']) if val == 0]

        if not pos_indices or not neg_indices:
            continue

        neg_texts = [passages['passage_text'][i] for i in neg_indices]

        for p_idx in pos_indices:
            out_queries.append(query)
            out_positives.append(passages['passage_text'][p_idx])
            out_negatives.append(neg_texts)

    return {
        "query": out_queries,
        "positive": out_positives,
        "negatives": out_negatives,
    }

class MSMarcoDataset(Dataset):
    def __init__(self, split="train"):
        raw = load_dataset("microsoft/ms_marco", "v1.1")
        self.data = raw[split].map(
            preprocess_expanded_negatives,
            batched=True,
            remove_columns=raw[split].column_names,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negatives": item["negatives"],  # list of strings
        }


def collate_fn(batch):
    return {
        "queries":    [item["query"]    for item in batch],
        "positives": [item["positive"] for item in batch],
        "negative_lists": [item["negatives"] for item in batch],  # list of lists
    }


if __name__ == "__main__":
    dataset = MSMarcoDataset(split="train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        queries   = batch["queries"]      # list of 32 strings
        positives = batch["positives"]   # list of 32 strings
        negatives = batch["negative_lists"]  # list of 32 lists-of-strings
        print(queries[0])
        print(positives[0])
        print(negatives[0][0])
        break