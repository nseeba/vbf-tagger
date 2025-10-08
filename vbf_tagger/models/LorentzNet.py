import math
import torch
from torch import nn
from typing import Tuple
import lightning as L
import torch.optim as optim
import torch.nn.functional as F


class LGEB(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 0,
        dropout: float = 0.0,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ) -> None:
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # dims for Minkowski norm & inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        if not last_layer:
            self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

        self.last_layer = last_layer

    def m_model(self, hi: torch.Tensor, hj: torch.Tensor, norms: torch.Tensor, dots: torch.Tensor) -> torch.Tensor:
        out = torch.cat([hi, hj, norms, dots], dim=-1)
        out = out.view(-1, out.size(dim=-1))
        out = self.phi_e(out)
        out = out.reshape(self.batchsize, -1, out.size(dim=-1))
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h: torch.Tensor, segment_ids: torch.Tensor, m: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        agg = unsorted_segment_sum(m, segment_ids, num_segments=self.n_particles)
        agg = torch.cat([h, agg, node_attr], dim=-1)
        agg = agg.view(-1, agg.size(dim=-1))
        agg = self.phi_h(agg)
        agg = agg.reshape(self.batchsize, -1, agg.size(dim=-1))
        out = h + agg
        return out

    def x_model(self, x: torch.Tensor, segment_ids: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "phi_x")
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100.0, max=+100.0)
        agg = unsorted_segment_mean(trans, segment_ids, num_segments=self.n_particles)
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(
        self, edgei: torch.Tensor, edgej: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edgei = edgei.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xi = torch.gather(x, 1, edgei)
        edgej = edgej.unsqueeze(dim=2).expand(-1, -1, x.size(dim=2))
        xj = torch.gather(x, 1, edgej)
        x_diff = xi - xj
        norms = normsq4(x_diff).unsqueeze(dim=-1)
        dots = dotsq4(xi, xj).unsqueeze(dim=-1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, edgei: torch.Tensor, edgej: torch.Tensor, node_attr: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.batchsize = h.size(dim=0)
        assert x.size(dim=0) == self.batchsize
        assert edgei.size(dim=0) == self.batchsize
        assert edgej.size(dim=0) == self.batchsize
        segment_ids = edgei

        self.n_particles = h.size(dim=-2)

        norms, dots, x_diff = self.minkowski_feats(edgei, edgej, x)

        hi = torch.gather(h, 1, edgei.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        hj = torch.gather(h, 1, edgej.unsqueeze(dim=2).expand(-1, -1, h.size(dim=2)))
        m = self.m_model(hi, hj, norms, dots)  # [B*N, hidden]

        # print("x:", x.shape, "agg:", agg.shape, "c_weight:", self.c_weight.shape)

        if not self.last_layer:
            x = self.x_model(x, segment_ids, x_diff, m)
        h = self.h_model(h, segment_ids, m, node_attr)
        return h, x, m


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    # CV: the following einsum operator gives the same as the operation
    #       'result = segment_ids @ data'
    #     but makes it more explicit how to multiply tensors in three dimensions
    result = torch.einsum("ijk,ikl->ijl", segment_ids, data)
    return result


def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`."""
    segment_ids = torch.nn.functional.one_hot(segment_ids, num_segments)
    segment_ids = torch.transpose(segment_ids, -2, -1).float()
    # CV: the following einsum operators give the same as the operations
    #       'result = segment_ids @ data' and 'count = segment_ids @ torch.ones_like(data)',
    #     but make it more explicit how to multiply tensors in three dimensions
    result = torch.einsum("ijk,ikl->ijl", segment_ids, data)
    count = torch.einsum("ijk,ikl->ijl", segment_ids, torch.ones_like(data))
    result = result / count.clamp(min=1)
    return result


def normsq4(p: torch.Tensor) -> torch.Tensor:
    r"""Minkowski square norm
    |p|^2 = p[0]^2 - p[1]^2 - p[2]^2 - p[3]^2
    """
    psq = torch.pow(p, 2)
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""Minkowski inner product
    <p,q> = p[0]q[0] - p[1]q[1] - p[2]q[2] - p[3]q[3]
    """
    psq = p * q
    result = 2 * psq[..., 0] - psq.sum(dim=-1)
    return result


def psi(p: torch.Tensor) -> torch.Tensor:
    """psi(p) = Sgn(p) * log(|p| + 1)"""
    result = torch.sign(p) * torch.log(torch.abs(p) + 1)
    return result


class LorentzNet(nn.Module):
    r"""Implementation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    """

    def __init__(
        self,
        n_scalar: int,
        n_hidden: int,
        n_class: int = 2,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        verbosity: int = 0,
    ) -> None:
        if verbosity >= 2:
            print("<LorentzNet::LorentzNet>:")
            print(" n_scalar = %i" % n_scalar)
            print(" n_hidden = %i" % n_hidden)
            print(" n_class = %i" % n_class)
            print(" n_layers = %i" % n_layers)
            print(" c_weight = %1.3f" % c_weight)
            print(" dropout = %1.2f" % dropout)
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    self.n_hidden,
                    self.n_hidden,
                    self.n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        # # old original LN
        # self.graph_dec = nn.Sequential(
        #     nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.n_hidden, n_class)
        # )  # classification

        #classifier per jet
        self.jet_classifier = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.n_hidden, n_class)
        )

        self.verbosity = verbosity

        self.beams_mass = 1
        self.beam1_p4 = [math.sqrt(1 + self.beams_mass**2), 0.0, 0.0, +1.0]
        self.beam2_p4 = [math.sqrt(1 + self.beams_mass**2), 0.0, 0.0, -1.0]

    def forward(self, cand_features, cand_kinematics, cand_mask):
        # shapes: cand_features (N, C, P), cand_kinematics (N, 4, P), cand_mask (N, 1, P)
        cand_kinematics = torch.swapaxes(cand_kinematics, 1, 2)  # (N,P,4)
        cand_features = torch.swapaxes(cand_features, 1, 2)      # (N,P,C)
        cand_mask = torch.swapaxes(cand_mask, 1, 2)              # (N,P,1)

        N, P, C = cand_features.shape

        # add beam particles (same as before)
        beams_kinematics_tensor = torch.tensor(
            [[self.beam1_p4, self.beam2_p4]], dtype=torch.float32
        ).to(cand_kinematics.device).expand(N, 2, 4)
        cand_kinematics = torch.cat([beams_kinematics_tensor, cand_kinematics], dim=1)

        beam_features = torch.nn.functional.pad(
            torch.tensor([[+1.0, -1.0]]), (0, 0, 0, C-1)
        ).to(cand_kinematics.device).expand(N, C, 2).swapaxes(1, 2)
        cand_features = torch.cat([beam_features, cand_features], dim=1)

        beams_mask = torch.ones((N, 2, 1), dtype=torch.float32).to(cand_kinematics.device)
        cand_mask = torch.cat([beams_mask, cand_mask], dim=1)

        # embed scalars
        h = self.embedding(cand_features)

        # build graph edges
        n_particles = cand_kinematics.size(1)
        edges = torch.ones(n_particles, n_particles, dtype=torch.long, device=h.device)
        edges = torch.triu(edges, diagonal=1) + torch.tril(edges, diagonal=-1)
        edges = torch.nonzero(edges).T
        edgei, edgej = edges[0].unsqueeze(0).expand(N, -1), edges[1].unsqueeze(0).expand(N, -1)

        # LGEB layers
        for layer in self.LGEBs:
            h, cand_kinematics, _ = layer(h, cand_kinematics, edgei, edgej, node_attr=cand_features)

        # mask padded jets
        h = h * cand_mask

        # classifier per jet → (N, P, n_class)
        pred = self.jet_classifier(h)

        return pred, cand_mask

class classification(L.LightningModule):
    def __init__(self, name: str, hyperparameters: dict, checkpoint: dict = None):
        super().__init__()
        self.name = name
        self.hparams.update(hyperparameters)
        self.checkpoint = checkpoint

        self.model = LorentzNet(
            n_scalar=self.hparams["n_scalar"],
            n_hidden=self.hparams["n_hidden"],
            n_class=self.hparams["n_class"],
            n_layers=self.hparams["n_layers"],
            c_weight=self.hparams.get("c_weight", 1e-3),
            dropout=self.hparams.get("dropout", 0.0),
        )
        self.lr = self.hparams["lr"]

    def forward(self, batch):
        cand_features, cand_kinematics, cand_mask, targets = batch
        preds = self.model(cand_features, cand_kinematics, cand_mask)
        if isinstance(preds, tuple):
            preds = preds[0]
        # print("Predictions tensor shape:", preds.shape)
        return preds, targets, cand_mask

    # def compute_loss_and_acc(self, preds, targets, mask):

    #     preds = preds[:, 2:, :]  # now shape is (B, P, n_class)
    #     # flatten everything
    #     preds = preds.reshape(-1, preds.size(-1))
    #     # print("shape of masked preds", preds.shape)
    #     targets = targets.view(-1).long()    
    #     # print("shape of targets", targets.shape)        # (N*P,)
    #     mask = mask.view(-1).bool()                  # (N*P,)
    #     # print("shape of mask", mask.shape)

    #     # apply mask
    #     preds = preds[mask]
    #     targets = targets[mask]

    #     loss = F.cross_entropy(preds, targets)
    #     acc = (preds.argmax(dim=-1) == targets).float().mean()
    #     return loss, acc

    def compute_loss_and_acc(self, preds, targets, mask):
        preds = preds[:, 2:, :]  # shape (B, P, n_class)
        preds = preds.reshape(-1, preds.size(-1))
        targets = targets.view(-1).long()
        mask = mask.view(-1).bool()

        preds = preds[mask]
        targets = targets[mask]

        # class balancing
        pos_weight = self.hparams.get("pos_weight", None)
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight], device=preds.device)
            loss = F.cross_entropy(preds, targets, weight=weight)
        else:
            loss = F.cross_entropy(preds, targets)

        acc = (preds.argmax(dim=-1) == targets).float().mean()
        return loss, acc


    def training_step(self, batch, batch_idx):
        preds, targets, mask = self.forward(batch)
        loss, acc = self.compute_loss_and_acc(preds, targets, mask)

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, targets, mask = self.forward(batch)
        loss, acc = self.compute_loss_and_acc(preds, targets, mask)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}







    # # Old original LN
    # def forward(self, cand_features, cand_kinematics, cand_mask) -> torch.Tensor:
    #     # cand_features: (N=num_batches, C=num_features, P=num_particles)
    #     # cand_kinematics_pxpypze: (N, 4, P) [px,py,pz,energy]
    #     # cand_mask: (N, 1, P) -- real particle = 1, padded = 0

    #     cand_kinematics = torch.swapaxes(cand_kinematics, 1, 2) #(N, 4, P) -> (N, P, 4)
    #     cand_features = torch.swapaxes(cand_features, 1, 2) #(N, C, P) -> (N, P, C)
    #     cand_mask = torch.swapaxes(cand_mask, 1, 2) #(N, 1, P) -> (N, P, 1)

    #     batch_size = cand_features.shape[0]
    #     num_features = cand_features.shape[2]

    #     #add two fake "beam" particles are required by LorentzNet
    #     #(N, P, 4) -> (N, P+2, 4)
    #     beams_kinematics_tensor = torch.tensor([[self.beam1_p4, self.beam2_p4]], dtype=torch.float32).to(cand_kinematics.device).expand(batch_size, 2, 4)
    #     cand_kinematics = torch.concatenate([beams_kinematics_tensor, cand_kinematics], axis=1)

    #     #(N, P, C) -> (N, P+2, C)
    #     beam_features = torch.nn.functional.pad(torch.tensor([[+1.0, -1.0]]), (0, 0, 0, num_features-1)).to(cand_kinematics.device).expand(batch_size, num_features, 2).swapaxes(1,2)
    #     cand_features = torch.concatenate([beam_features, cand_features], axis=1)

    #     #(N, P, 1) -> (N, P+2, 1)
    #     beams_mask = torch.ones((batch_size, 2, 1), dtype=torch.float32).to(cand_kinematics)
    #     cand_mask = torch.concatenate([beams_mask, cand_mask], axis=1)

    #     #embed the per-particle non Lorentz invariant quantities (scalars)
    #     h = self.embedding(cand_features)

    #     #create particle-to-particle "edges" within each jet with all-to-all connections
    #     n_particles = cand_kinematics.size(dim=1)
    #     edges = torch.ones(n_particles, n_particles, dtype=torch.long, device=h.device)
    #     edges_above_diag = torch.triu(edges, diagonal=1)
    #     edges_below_diag = torch.tril(edges, diagonal=-1)
    #     edges = torch.add(edges_above_diag, edges_below_diag)
    #     edges = torch.nonzero(edges)
    #     edges = torch.swapaxes(edges, 0, 1)
    #     edgei = torch.unsqueeze(edges[0], dim=0)
    #     edgei = edgei.expand(h.size(dim=0), -1)
    #     edgej = torch.unsqueeze(edges[1], dim=0)
    #     edgej = edgej.expand(h.size(dim=0), -1)

    #     for i in range(self.n_layers):
    #         h, cand_kinematics, _ = self.LGEBs[i].forward(h, cand_kinematics, edgei, edgej, node_attr=cand_features)
    #     h = h * cand_mask
    #     h = h.view(-1, n_particles, self.n_hidden)
    #     h = torch.mean(h, dim=1)
    #     pred = self.graph_dec(h)
    #     result = pred.squeeze(0)
    #     return result