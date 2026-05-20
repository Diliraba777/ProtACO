"""Tests for ProtACO dataset loading and collation."""

import pytest
import torch
from torch.utils.data import DataLoader

from protaco.dataset import ProteinEmbeddingDataset, collate_fn


def test_dataset_reads_embeddings_from_lmdb(tiny_embedding_data):
    dataset = ProteinEmbeddingDataset(
        tiny_embedding_data["csv"],
        tiny_embedding_data["lmdb"],
    )

    embedding, score = dataset[0]

    assert len(dataset) == 3
    assert embedding.shape == (tiny_embedding_data["lengths"][0], tiny_embedding_data["embedding_dim"])
    assert embedding.dtype == torch.float32
    assert score.item() == pytest.approx(tiny_embedding_data["scores"][0])


def test_collate_fn_pads_variable_length_embeddings(tiny_embedding_data):
    dataset = ProteinEmbeddingDataset(
        tiny_embedding_data["csv"],
        tiny_embedding_data["lmdb"],
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)

    embeddings, scores = next(iter(loader))

    assert embeddings.shape == (3, max(tiny_embedding_data["lengths"]), tiny_embedding_data["embedding_dim"])
    assert scores.shape == (3,)
    assert torch.all(embeddings[2, tiny_embedding_data["lengths"][2] :, :] == 0)


def test_dataset_reports_missing_columns(tmp_path, tiny_embedding_data):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("id,wrong_target\nprotein_001,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required column"):
        ProteinEmbeddingDataset(bad_csv, tiny_embedding_data["lmdb"])
