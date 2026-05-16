import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef


class TransformerEncoderRegressor(pl.LightningModule):
    """
    Transformer encoder regressor for protein embeddings.

    The model pools token-level embeddings into a fixed-size representation,
    projects it into a tmvec-style structure vector, and predicts a non-negative
    regression score. Training combines MSLE with a pairwise difference loss.
    """

    def __init__(
        self,
        d_model=1024,
        nhead=8,
        num_layers=2,
        dim_feedforward=2048,
        out_dim=512,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.structure_proj = nn.Linear(d_model, out_dim)

        self.final_mlp = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.train_corr = PearsonCorrCoef()
        self.val_corr = PearsonCorrCoef()
        self.lr = lr

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Protein embeddings with shape [batch, seq_len, d_model].
            mask: Boolean padding mask with shape [batch, seq_len].
                True marks padded positions.
        """
        out = self.encoder(x, src_key_padding_mask=mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            out = out.masked_fill(mask_expanded, 0.0)

            valid_counts = torch.logical_not(mask).sum(dim=1).unsqueeze(1)
            valid_counts = valid_counts.clamp(min=1)
            pooled = out.sum(dim=1) / valid_counts
        else:
            pooled = out.mean(dim=1)

        pooled = self.dropout(pooled)
        structure_vec = self.structure_proj(pooled)
        pred = self.final_mlp(structure_vec).squeeze(-1)

        # MSLE requires non-negative predictions.
        return F.softplus(pred)

    def pairwise_loss(self, preds, targets):
        """
        Preserve relative differences between samples in log space.
        """
        log_preds = torch.log1p(preds)
        log_targets = torch.log1p(targets)

        diff_preds = log_preds.unsqueeze(1) - log_preds.unsqueeze(0)
        diff_targets = log_targets.unsqueeze(1) - log_targets.unsqueeze(0)
        return F.mse_loss(diff_preds, diff_targets)

    def _prepare_batch(self, batch):
        x, y = batch
        y = y.float()

        if x.dim() == 2:
            x_in = x.unsqueeze(1)
            mask = torch.zeros(
                (x_in.size(0), x_in.size(1)),
                dtype=torch.bool,
                device=x_in.device,
            )
        else:
            x_in = x
            mask = x_in.abs().sum(-1) == 0

        return x_in, y, mask

    def training_step(self, batch, batch_idx):
        x, y, mask = self._prepare_batch(batch)
        y_hat = self(x, mask)

        msle_loss = F.mse_loss(torch.log1p(y_hat), torch.log1p(y))
        pair_loss = self.pairwise_loss(y_hat, y)
        loss = msle_loss + pair_loss
        loss_ratio = pair_loss / (msle_loss + 1e-8)

        corr = self.train_corr(y_hat, y)

        self.log("t_loss", loss, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("t_corr", corr, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        self.log("train_loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_corr", corr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss_ratio", loss_ratio, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = self._prepare_batch(batch)
        y_hat = self(x, mask)

        val_msle = F.mse_loss(torch.log1p(y_hat), torch.log1p(y))
        corr = self.val_corr(y_hat, y)

        self.log("v_loss", val_msle, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("v_corr", corr, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        self.log("val_loss", val_msle, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_corr", corr, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return val_msle

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        max_epochs = self.trainer.max_epochs if self.trainer else 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return [optimizer], [scheduler]
