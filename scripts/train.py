"""Train the ProtACO regressor."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from protaco.dataset import ProteinEmbeddingDataset, collate_fn
from protaco.model import TransformerEncoderRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ProtACO regressor.")

    parser.add_argument("--csv", required=True, help="Training metadata CSV.")
    parser.add_argument("--lmdb", required=True, help="LMDB path containing precomputed embeddings.")
    parser.add_argument("--id-col", default="id", help="Protein ID column in the CSV.")
    parser.add_argument("--target-col", default="CO_score", help="Regression target column in the CSV.")
    parser.add_argument("--output-dir", default="runs/protaco", help="Directory for logs, checkpoints, and plots.")
    parser.add_argument("--run-name", default="protaco", help="TensorBoard run name.")
    parser.add_argument("--version", type=int, default=None, help="Run version. Auto-incremented when omitted.")
    parser.add_argument("--min-version", type=int, default=0, help="Minimum auto-generated version number.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--out-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--accelerator", default="auto", help="Lightning accelerator, e.g. auto, cpu, gpu.")
    parser.add_argument("--devices", default="auto", help="Lightning devices, e.g. auto, 1, 4, or 0,1.")
    parser.add_argument("--strategy", default="auto", help="Lightning strategy, e.g. auto or ddp.")
    parser.add_argument("--eval-device", default="auto", help="Device used for post-training validation plots.")

    return parser.parse_args()


def parse_devices(value):
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        return value


def resolve_eval_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def next_version(log_root, min_version):
    if not log_root.exists():
        return min_version

    existing_versions = []
    for path in log_root.iterdir():
        if path.is_dir() and path.name.startswith("version_"):
            try:
                existing_versions.append(int(path.name.split("_", 1)[1]))
            except ValueError:
                pass

    if not existing_versions:
        return min_version
    return max(max(existing_versions) + 1, min_version)


def build_loaders(args):
    dataset = ProteinEmbeddingDataset(
        args.csv,
        args.lmdb,
        id_column=args.id_col,
        target_column=args.target_col,
    )

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    val_size = max(1, int(args.val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_model(args):
    return TransformerEncoderRegressor(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        out_dim=args.out_dim,
        dropout=args.dropout,
        lr=args.lr,
    )


def save_validation_outputs(best_model_path, val_ds, args, plot_dir, version):
    eval_device = resolve_eval_device(args.eval_device)
    print(f"Loading best checkpoint for validation plots: {best_model_path}")

    best_model = TransformerEncoderRegressor.load_from_checkpoint(
        best_model_path,
        map_location=eval_device,
    )
    best_model.to(eval_device)
    best_model.eval()

    plot_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in plot_loader:
            x = x.to(eval_device)
            mask = (x.abs().sum(-1) == 0) if x.dim() == 3 else None
            y_hat = best_model(x, mask=mask)
            y_pred.extend(y_hat.cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())

    val_indices = val_ds.indices
    errors = np.array(y_true) - np.array(y_pred)

    df_res = pd.DataFrame(
        {
            "original_index": val_indices,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": errors,
            "abs_error": np.abs(errors),
        }
    ).sort_values(by="abs_error", ascending=False)

    csv_save_path = plot_dir / "val_predictions_detailed.csv"
    df_res.to_csv(csv_save_path, index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal y=x")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Pred vs True (Val) - Version {version}")
    plt.legend()
    plt.tight_layout()
    pred_plot_path = plot_dir / "pred_vs_true.png"
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="r", linestyle="--")
    plt.title("Error Distribution")
    plt.xlabel("Error (True - Pred)")

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, errors, alpha=0.5, s=10, c="purple")
    plt.axhline(0, color="r", linestyle="--")
    plt.title("Residuals vs True Values")
    plt.xlabel("True Values")
    plt.ylabel("Error")

    plt.tight_layout()
    error_plot_path = plot_dir / "error_analysis.png"
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print("Validation outputs saved:")
    print(f"  CSV: {csv_save_path}")
    print(f"  Pred vs true plot: {pred_plot_path}")
    print(f"  Error analysis plot: {error_plot_path}")


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    log_root = log_dir / args.run_name
    version = args.version if args.version is not None else next_version(log_root, args.min_version)

    checkpoint_dir = output_dir / "checkpoints" / f"version_{version}"
    plot_dir = output_dir / "plots" / f"version_{version}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run version: {version}")
    print(f"Logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Plots: {plot_dir}")

    _, val_ds, train_loader, val_loader = build_loaders(args)
    model = build_model(args)

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=args.run_name,
        version=version,
    )

    best_ckpt = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    last_ckpt = ModelCheckpoint(dirpath=str(checkpoint_dir), save_last=True)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
        logger=logger,
        callbacks=[best_ckpt, last_ckpt, early_stop],
    )

    trainer.fit(model, train_loader, val_loader)

    if trainer.global_rank == 0:
        best_model_path = best_ckpt.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            save_validation_outputs(best_model_path, val_ds, args, plot_dir, version)
            print("Training complete.")
            print(f"TensorBoard: tensorboard --logdir {log_dir}")
            print(f"Best checkpoint: {best_model_path}")
        else:
            print("Warning: Best checkpoint not found. Skipping validation plots.")


if __name__ == "__main__":
    main()
