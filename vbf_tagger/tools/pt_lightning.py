import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

class PTLightningModule(L.LightningModule):
    def __init__(self, model, lr=1e-4, pos_weight=1.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()

    def forward(self, batch):
        x = batch["points"]          # (B, P, F)
        coords = batch["points_xyz"]     # (B, P, 4)
        mask = batch["points_mask"]  # (B, P)

        # ParticleTransformer expects: (P, N, C) for x and (N, C, P) for BatchNorm
        # 1. Transpose features: (B, P, F) -> (B, F, P)
        x = x.transpose(1, 2).contiguous() 
        
        # 2. Transpose coordinates: (B, P, 4) -> (B, 4, P)
        coords = coords.transpose(1, 2).contiguous()  
        
        # 3. Unsqueeze mask: (B, P) -> (B, 1, P)
        mask = mask.unsqueeze(1)
        
        return self.model(x, coords, mask)

    def _masked_metrics(self, preds, targets, mask):
        # flatten only valid jets
        preds = preds[mask]
        targets = targets[mask]
        return preds, targets

    def training_step(self, batch, batch_idx):
        logits = self(batch)               # (B, P, 1)
        logits = logits.squeeze(-1)

        targets = batch["labels"]          # (B, P)
        mask = batch["points_mask"]        # (B, P)

        # Mask padded jets
        logits_masked, targets_masked = self._masked_metrics(
            logits, targets, mask
        )

        loss = self.loss_fn(logits_masked, targets_masked.float())
        preds = torch.sigmoid(logits_masked)

        acc = self.train_acc(preds, targets_masked.int())

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch).squeeze(-1)
        targets = batch["labels"]
        mask = batch["points_mask"]

        logits_masked, targets_masked = self._masked_metrics(
            logits, targets, mask
        )
        preds = torch.sigmoid(logits_masked)

        loss = self.loss_fn(logits_masked, targets_masked.float())
        acc = self.val_acc(preds, targets_masked.int())
        auc = self.val_auc(preds, targets_masked.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Define the ReduceLROnPlateau scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',         # Monitor a metric we want to minimize (val_loss)
            factor=0.5,         # Reduce LR by 50% when plateau is reached
            patience=3,         # Wait for 3 epochs without val_loss improvement before reducing LR
            min_lr=1e-6,        # Set a floor for the learning rate
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Tell Lightning to monitor this logged metric
                "interval": "epoch",    # Check the scheduler after every epoch
                "frequency": 1,         # Check every epoch
            },
        }
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr)
