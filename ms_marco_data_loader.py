import os
from torch.utils.data import Dataset, DataLoader


class MSMarcoTriplets(Dataset):
    def __init__(self, path):
        self.path = path
        self.offsets = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8").strip()
        query, positive, negative = line.split("\t")
        return {"query": query, "positive": positive, "negative": negative}


def collate_fn(batch):
    return {
        "queries": [item["query"] for item in batch],
        "positives": [item["positive"] for item in batch],
        "negatives": [item["negative"] for item in batch],
    }


if __name__ == "__main__":
    dataset = MSMarcoTriplets("data/triples.train.small.tsv")
    print(f"Total triplets: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        print(batch["queries"][0])
        print(batch["positives"][0])
        print(batch["negatives"][0])
        break
