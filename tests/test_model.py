"""Tests for the ProtACO model."""

import torch

from protaco.model import TransformerEncoderRegressor


def build_tiny_model():
    return TransformerEncoderRegressor(
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        out_dim=8,
        dropout=0.0,
        lr=1e-3,
    )


def test_forward_returns_non_negative_predictions():
    model = build_tiny_model()
    x = torch.randn(3, 5, 16)
    mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]
    )

    preds = model(x, mask)

    assert preds.shape == (3,)
    assert torch.all(preds >= 0)


def test_training_step_returns_scalar_loss():
    model = build_tiny_model()
    x = torch.randn(3, 5, 16)
    x[1, 3:, :] = 0
    x[2, 2:, :] = 0
    y = torch.tensor([0.12, 0.35, 0.08], dtype=torch.float32)

    loss = model.training_step((x, y), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pairwise_loss_is_zero_when_predictions_match_targets():
    model = build_tiny_model()
    values = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)

    loss = model.pairwise_loss(values, values)

    assert loss.item() == 0.0
