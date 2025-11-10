# vbf_tagger/models/MLPClassifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import io


class MLPClassifier(L.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3, pos_weight=1.0):
        super().__init__()
        self.save_hyperparameters()

        # Define MLP architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        # Buffers for validation aggregation
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        """Forward pass"""
        return self.model(x).squeeze(-1)

    def configure_optimizers(self):
        """Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _bce_loss(self, logits, targets):
        """Binary cross-entropy with optional positive weighting"""
        pos_weight = torch.tensor(self.hparams.pos_weight, device=self.device)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)

    def training_step(self, batch, batch_idx):
        """Run one training step"""
        x, y = batch
        logits = self(x)
        loss = self._bce_loss(logits, y)

        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Run one validation step"""
        x, y = batch
        logits = self(x)
        preds = torch.sigmoid(logits)

        loss = self._bce_loss(logits, y)
        acc = ((preds > 0.5) == y).float().mean()

        # Store for ROC/AUC computation
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return  # no data, sanity check

        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        self.val_preds.clear()
        self.val_targets.clear()

        # Compute ROC curve and AUC
        try:
            fpr, tpr, _ = roc_curve(targets.numpy(), preds.numpy())
            roc_auc = auc(fpr, tpr)
        except ValueError:
            roc_auc = float("nan")

        # Log AUC to progress bar and logger
        self.log("val_auc", roc_auc, prog_bar=True, on_epoch=True)

        # --- Log ROC curve image to Comet (Lightning-friendly) ---
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(tpr, 1 - fpr, label=f"AUC={roc_auc:.3f}", color="red")
            ax.set_xlabel("Signal efficiency (TPR)")
            ax.set_ylabel("Background rejection (1-FPR)")
            ax.legend()
            ax.grid(alpha=0.3)

            # Save ROC plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            # Correct handling for Lightning's CometLogger
            if isinstance(self.logger, list):
                # If multiple loggers, pick the Comet one
                comet_loggers = [l for l in self.logger if l.__class__.__name__ == "CometLogger"]
                if comet_loggers:
                    comet_logger = comet_loggers[0]
                    comet_logger.experiment.log_image(buf, name=f"ROC_Epoch_{self.current_epoch}")
            elif hasattr(self.logger, "experiment"):
                exp = self.logger.experiment
                if hasattr(exp, "log_image"):
                    exp.log_image(buf, name=f"ROC_Epoch_{self.current_epoch}")
            elif hasattr(self.logger, "log_image"):
                self.logger.log_image(buf, name=f"ROC_Epoch_{self.current_epoch}")

            plt.close(fig)
        except Exception as e:
            print(f"[Warning] Could not log ROC curve: {e}")

        # # Log ROC curve to Comet (works with Lightning's CometLogger)
        # if self.logger is not None:
        #     try:
        #         fig, ax = plt.subplots(figsize=(6, 5))
        #         ax.plot(tpr, 1 - fpr, label=f"AUC={roc_auc:.3f}", color="red")
        #         ax.set_xlabel("Signal efficiency (TPR)")
        #         ax.set_ylabel("Background rejection (1-FPR)")
        #         ax.legend()
        #         ax.grid(alpha=0.3)

        #         # Save to buffer
        #         buf = io.BytesIO()
        #         plt.savefig(buf, format="png")
        #         buf.seek(0)

        #         # Handle both CometLogger and plain Experiment
        #         if hasattr(self.logger, "experiment"):
        #             exp = self.logger.experiment
        #             if hasattr(exp, "log_figure"):
        #                 exp.log_figure(figure_name="ROC_Curve", figure=buf)
        #             elif hasattr(exp, "log_image"):
        #                 exp.log_image(image_data=buf, name="ROC_Curve")
        #         elif hasattr(self.logger, "log_figure"):
        #             self.logger.log_figure(figure_name="ROC_Curve", figure=buf)
        #         elif hasattr(self.logger, "log_image"):
        #             self.logger.log_image(image_data=buf, name="ROC_Curve")

        #         plt.close(fig)
        #     except Exception as e:
        #         print(f"[Warning] Could not log ROC curve: {e}")
