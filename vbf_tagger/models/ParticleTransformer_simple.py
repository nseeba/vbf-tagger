# vbf_tagger/models/ParticleTransformer.py
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ParticleTransformer(L.LightningModule):
    """
    Lightweight, faithful ParticleTransformer-style tagger:
     - inputs: (B, max_jets, n_features)
     - mask:   (B, max_jets) boolean (True = real jet)
     - labels: (B, max_jets) float {0,1}

    Produces per-jet logits (no softmax) and uses BCEWithLogits on masked tokens.
    Logs train/val loss, accuracy, and val_auc; pushes ROC image to Comet if available.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # input projection (per-jet)
        self.input_proj = nn.Linear(n_features, d_model)

        # optional learned "jet index" embedding (helps give token order)
        self.jet_pos_emb = nn.Embedding(512, d_model)  # max_jets <= 512; safe default

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification head per jet (logit)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # buffers for validation epoch aggregation
        self.val_preds = []
        self.val_targets = []

        # learning rate & pos_weight
        self.lr = lr
        self.register_buffer("pos_weight_buffer", torch.tensor(pos_weight, dtype=torch.float32))

    def forward(self, x, mask=None):
        """
        x: (B, T, F)
        mask: (B, T) bool, True = real jet
        returns logits: (B, T)
        """

        B, T, F = x.shape
        # project
        h = self.input_proj(x)                     # (B, T, d_model)

        # add small learned positional embedding (index by jet position 0..T-1)
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.jet_pos_emb(idx)              # (B, T, d_model)

        # transformer expects (T, B, d_model)
        h = h.transpose(0, 1)                      # (T, B, d_model)

        # build src_key_padding_mask for Transformers: True = pad token -> mask out
        if mask is None:
            src_key_padding_mask = None
        else:
            # mask: True = real jet. transformer expects True for positions that should be ignored
            src_key_padding_mask = ~mask           # (B, T) : True = padding (to ignore)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (T, B, d_model)

        h = h.transpose(0, 1)                      # (B, T, d_model)
        logits = self.head(h).squeeze(-1)          # (B, T)
        return logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return opt

    def _bce_loss_masked(self, logits, targets, mask):
        """
        logits: (B, T), targets: (B, T), mask: (B, T) bool
        compute BCEWithLogits over masked positions. apply pos_weight.
        """
        pos_weight = self.pos_weight_buffer.to(logits.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        loss_per_elem = loss_fn(logits, targets)
        # mask selects real jets
        masked = loss_per_elem * mask.float()
        # avoid division by zero if no masked elements: use sum / num_real
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom

    def training_step(self, batch, batch_idx):
        X, mask, y = batch  # X: (B,T,F), mask (B,T), y (B,T)
        logits = self(X, mask)
        loss = self._bce_loss_masked(logits, y, mask)

        preds = torch.sigmoid(logits)
        # compute accuracy on masked tokens
        pred_labels = (preds > 0.5).float()
        acc = ((pred_labels == y) * mask).sum().float() / mask.sum().clamp_min(1.0)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, mask, y = batch
        logits = self(X, mask)
        loss = self._bce_loss_masked(logits, y, mask)

        preds = torch.sigmoid(logits)

        # collect masked preds/targets to CPU lists for ROC
        masked_preds = preds.detach().cpu()[mask.detach().cpu()]
        masked_targets = y.detach().cpu()[mask.detach().cpu()]

        if masked_preds.numel() > 0:
            self.val_preds.append(masked_preds)
            self.val_targets.append(masked_targets)

        # accuracy
        pred_labels = (preds > 0.5).float()
        acc = ((pred_labels == y) * mask).sum().float() / mask.sum().clamp_min(1.0)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # if nothing collected
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
        self.val_preds.clear()
        self.val_targets.clear()

        # ROC/AUC
        try:
            fpr, tpr, _ = roc_curve(targets, preds)
            roc_auc = auc(fpr, tpr)
        except ValueError:
            roc_auc = float("nan")

        # log scalar
        self.log("val_auc", float(roc_auc), prog_bar=True, on_epoch=True)

        # log ROC image to Comet (compatible with Lightning CometLogger)
        try:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(tpr, 1 - fpr, label=f"AUC={roc_auc:.4f}", color="tab:orange")
            ax.set_xlabel("Signal efficiency (TPR)")
            ax.set_ylabel("Background rejection (1-FPR)")
            ax.grid(alpha=0.3)
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            # Comet logger handling
            if isinstance(self.logger, list):
                comet_loggers = [l for l in self.logger if l.__class__.__name__ == "CometLogger"]
                if len(comet_loggers) > 0:
                    comet_loggers[0].experiment.log_image(buf, name=f"ROC_epoch_{self.current_epoch}")
            elif hasattr(self.logger, "experiment"):
                exp = self.logger.experiment
                if hasattr(exp, "log_image"):
                    exp.log_image(buf, name=f"ROC_epoch_{self.current_epoch}")
        except Exception as e:
            print(f"[Warning] Could not log ROC curve: {e}")
