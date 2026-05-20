"""Dataset utilities for ProtACO."""

import pickle

import lmdb
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset


class ProteinEmbeddingDataset(Dataset):
    def __init__(self, csv_file, lmdb_file, id_column="id", target_column="CO_score"):
        """
        Dataset for precomputed protein embeddings stored in LMDB.

        Args:
            csv_file: CSV file containing protein IDs and target values.
            lmdb_file: LMDB path containing pickled NumPy embeddings.
            id_column: CSV column used as the LMDB key.
            target_column: CSV column used as the regression target.
        """
        self.data = pd.read_csv(csv_file, sep=",")

        missing_columns = [col for col in (id_column, target_column) if col not in self.data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required column(s) in {csv_file}: {', '.join(missing_columns)}"
            )

        self.accessions = self.data[id_column].astype(str).tolist()
        self.scores = self.data[target_column].tolist()
        self.lmdb_file = str(lmdb_file)

        # Open lazily so DataLoader workers create their own LMDB handles.
        self.env = None

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_file,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        acc = self.accessions[idx]
        score = torch.tensor(self.scores[idx], dtype=torch.float32)

        with self.env.begin(write=False) as txn:
            emb_bytes = txn.get(acc.encode("utf-8"))
            if emb_bytes is None:
                raise KeyError(f"Embedding not found for accession: {acc}")
            emb = pickle.loads(emb_bytes)

        emb = torch.from_numpy(emb).float()
        return emb, score


def collate_fn(batch):
    """
    Pad variable-length token embeddings or stack protein-level embeddings.
    """
    embs, scores = zip(*batch)

    if embs[0].dim() == 1:
        embs = torch.stack(embs)
    else:
        embs = rnn_utils.pad_sequence(embs, batch_first=True)

    scores = torch.stack(scores)
    return embs, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test the protein embedding dataset.")
    parser.add_argument("--csv", required=True, help="CSV file with protein IDs and targets.")
    parser.add_argument("--lmdb", required=True, help="LMDB path with precomputed embeddings.")
    parser.add_argument("--id-col", default="id", help="Protein ID column in the CSV.")
    parser.add_argument("--target-col", default="CO_score", help="Regression target column in the CSV.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    dataset = ProteinEmbeddingDataset(
        args.csv,
        args.lmdb,
        id_column=args.id_col,
        target_column=args.target_col,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    for batch_embs, batch_scores in dataloader:
        print("Dataset smoke test succeeded.")
        print(f"Batch embeddings shape: {batch_embs.shape}")
        print(f"Batch scores shape: {batch_scores.shape}")
        break
