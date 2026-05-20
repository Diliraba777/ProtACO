"""Pytest fixtures for ProtACO smoke tests."""

import pickle

import lmdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tiny_embedding_data(tmp_path):
    ids = ["protein_001", "protein_002", "protein_003"]
    scores = [0.12, 0.35, 0.08]
    embedding_dim = 16
    lengths = [5, 8, 3]

    csv_path = tmp_path / "metadata.csv"
    lmdb_path = tmp_path / "embeddings.lmdb"

    pd.DataFrame({"id": ids, "CO_score": scores}).to_csv(csv_path, index=False)

    rng = np.random.default_rng(42)
    env = lmdb.open(str(lmdb_path), map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        for protein_id, length in zip(ids, lengths):
            embedding = rng.normal(size=(length, embedding_dim)).astype(np.float32)
            txn.put(protein_id.encode("utf-8"), pickle.dumps(embedding))
    env.close()

    return {
        "csv": csv_path,
        "lmdb": lmdb_path,
        "ids": ids,
        "scores": scores,
        "embedding_dim": embedding_dim,
        "lengths": lengths,
    }
