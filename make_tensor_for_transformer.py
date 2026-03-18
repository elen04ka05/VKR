import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("snp_batch_processing_results_20260130_170214/tokenized_sequences.csv")  # твой CSV
OUT_PATH = "genomic_mlm_data.pt"
BATCH_SIZE = 8

def build_vocab(tokens_column):
    vocab = {
        "[PAD]": 0,
        "[MASK]": 1
    }
    idx = 2

    for seq in tokens_column:
        for tok in seq.split(";"):
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1

    return vocab

vocab = build_vocab(df["tokens"])

pad_id  = vocab["[PAD]"]
mask_id = vocab["[MASK]"]
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

def tokens_to_ids(token_str, vocab):
    return [vocab[t] for t in token_str.split(";")]

def apply_masking(token_ids, masked_ids_str, mask_id):
    masked_ids = set(map(int, masked_ids_str.split(";")))

    input_ids = []
    labels = []

    for tid in token_ids:
        if tid in masked_ids:
            input_ids.append(mask_id)
            labels.append(tid)      # target = оригинальный ID
        else:
            input_ids.append(tid)
            labels.append(-100)     # ignore_index

    return input_ids, labels

class GenomicMLMDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.mask_id = vocab["[MASK]"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        token_ids = tokens_to_ids(row["tokens"], self.vocab)
        input_ids, labels = apply_masking(
            token_ids,
            row["masked_ids"],
            self.mask_id
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, labels, attention_mask = [], [], []

    for b in batch:
        L = len(b["input_ids"])
        pad_len = max_len - L

        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask.append(
            torch.cat([torch.ones(L), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids),          # [B, T]
        "labels": torch.stack(labels),                # [B, T]
        "attention_mask": torch.stack(attention_mask) # [B, T]
    }

dataset = GenomicMLMDataset(df, vocab)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ======================
# PROCESS ALL SAMPLES
# ======================
all_batches = []

for batch in loader:
    all_batches.append(batch)

torch.save(
    {
        "batches": all_batches,
        "vocab": vocab,
        "pad_id": pad_id,
        "mask_id": mask_id,
    },
    OUT_PATH
)

print(f"Saved {len(dataset)} samples to {OUT_PATH}")

print("input_ids:", batch["input_ids"].shape)
print("labels:", batch["labels"].shape)
print("attention_mask:", batch["attention_mask"].shape)

print("Example input_ids row:")
print(batch["input_ids"][0][:20])

print("Example labels row:")
print(batch["labels"][0][:20])

