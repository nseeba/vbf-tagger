# vbf_tagger/tools/data/pt_dataloader.py
import os
import glob
import numpy as np
import awkward as ak
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Dataset
from vbf_tagger.tools.data.general import initialize_p4
from omegaconf import DictConfig
from hydra import compose, initialize
from torch.utils.data._utils.collate import default_collate


def _compute_cartesian(pt, eta, phi):
    """
    Compute px, py, pz from (pt, eta, phi).
    px = pt * cos(phi)
    py = pt * sin(phi)
    pz = pt * sinh(eta)
    Returns shape (..., 3)
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack([px, py, pz], axis=-1)


def build_jet_tensors(data: ak.Array, features: list, max_jets: int = 14):
    """
    Build padded (n_events, max_jets, n_features) jet tensors for jet-level training.

    Returns:
        X:       float32 (B, P, F)
        coords:  float32 (B, P, 4)  # px, py, pz, E
        mask:    bool    (B, P)
        labels:  float32 (B, P)
    """

    jets_raw = data.TrainingJet
    jets_p4  = initialize_p4(data.TrainingJet)

    # ---------------------------
    # 1. Sort by descending pt
    # ---------------------------
    order = ak.argsort(jets_p4.pt, ascending=False)
    jets_raw = jets_raw[order]
    jets_p4  = jets_p4[order]

    # ---------------------------
    # padding helper
    # ---------------------------
    def pad(field):
        arr = ak.pad_none(field, max_jets, clip=True)
        arr = ak.fill_none(arr, 0)
        return ak.to_numpy(arr)

    # ---------------------------
    # 2. Derived safe features
    # ---------------------------
    pt_arr   = pad(jets_p4.pt).astype(np.float32)
    eta_arr  = pad(jets_p4.eta).astype(np.float32)
    phi_arr  = pad(jets_p4.phi).astype(np.float32)
    mass_arr = pad(jets_p4.mass).astype(np.float32)

    pt_safe   = np.clip(pt_arr,   1e-6, None)
    mass_safe = np.clip(mass_arr, 1e-6, None)

    derived = {
        "log_pt":   np.log(pt_safe),
        "sin_phi":  np.sin(phi_arr),
        "cos_phi":  np.cos(phi_arr),
        "log_mass": np.log(mass_safe),
    }

    # ---------------------------
    # 3. Build X
    # ---------------------------
    feat_list = []
    for f in features:
        if f in derived:
            arr = derived[f]
        elif hasattr(jets_p4, f):
            arr = pad(getattr(jets_p4, f))
        elif hasattr(jets_raw, f):
            arr = pad(getattr(jets_raw, f))
        else:
            raise ValueError(f"Feature '{f}' not found.")
        feat_list.append(arr[..., None])

    X = np.concatenate(feat_list, axis=-1).astype(np.float32)

    # ---------------------------
    # 4. Mask
    # ---------------------------
    pt_orig = pad(initialize_p4(data.TrainingJet).pt[order])
    mask = (pt_orig > 0)

    # ---------------------------
    # 5. Labels
    # ---------------------------
    labels = pad(jets_raw.isVBF).astype(np.float32)

    # ---------------------------
    # 6. Coords: (px,py,pz,E)
    # ---------------------------
    xyz3 = _compute_cartesian(pt_arr, eta_arr, phi_arr)  # (B,P,3)

    energy = np.sqrt(
        xyz3[..., 0]**2 + xyz3[..., 1]**2 + xyz3[..., 2]**2 + mass_safe**2
    )

    coords = np.concatenate([xyz3, energy[..., None]], axis=-1).astype(np.float32)
    # (B, P, 4)

    return X, coords, mask.astype(bool), labels


class JetDataModule(LightningDataModule):
    """
    Lightning DataModule producing batches as dicts compatible with ParticleTransformer.

    Each batch is a dict:
        {
          "points": Tensor[B, N, F],
          "points_mask": BoolTensor[B, N],
          "points_xyz": Tensor[B, N, 3],
          "labels": Tensor[B, N]
        }
    """
    def __init__(self, cfg: DictConfig, features: list = None, max_jets: int = 14):
        super().__init__()
        self.cfg = cfg
        self.features = features
        self.max_jets = max_jets
        self.batch_size = cfg.training.dataloader.batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # scaler (applies to features only)
        self.mean = None
        self.std = None
        self.pos_weight = 1.0

    def _get_files(self, dataset_keys, split_dir):
        dataset_paths = []
        for key in dataset_keys:
            base = self.cfg.dataset.datasets[key]
            path = os.path.join(base, split_dir)
            files = glob.glob(os.path.join(path, "*.parquet"))
            dataset_paths.extend(files)
        return sorted(dataset_paths)

    def setup(self, stage=None):
        # train/val
        if stage == "fit" or stage is None:
            train_files = self._get_files(self.cfg.dataset.train_dataset, self.cfg.dataset.train_dir)
            val_files   = self._get_files(self.cfg.dataset.val_dataset, self.cfg.dataset.val_dir)
            print(f" Found {len(train_files)} train files, {len(val_files)} val files")

            data_train = ak.from_parquet(train_files)
            X_train, coords_train, mask_train, y_train = build_jet_tensors(data_train, self.features, max_jets=self.max_jets)

            data_val = ak.from_parquet(val_files)
            X_val, coords_val, mask_val, y_val = build_jet_tensors(data_val, self.features, max_jets=self.max_jets)

            # compute scaler from TRAIN only (only real jets)
            mask_flat = mask_train.reshape(-1)
            X_flat = X_train.reshape(-1, X_train.shape[-1])
            valid = mask_flat
            if valid.sum() == 0:
                raise RuntimeError("No valid jets in training data.")
            mean = X_flat[valid].mean(axis=0)
            std = X_flat[valid].std(axis=0) + 1e-6
            self.mean = mean
            self.std = std

            # Save scaler for later inference
            scaler_path = os.path.join(self.cfg.training.models_dir, "scaler.npz")
            os.makedirs(self.cfg.training.models_dir, exist_ok=True)
            np.savez(scaler_path, mean=mean, std=std)
            print(f" Saved scaler → {scaler_path}")

            # apply normalization ONLY to X (features). coords left unchanged.
            X_train = (X_train - mean[None, None, :]) / std[None, None, :]
            X_val   = (X_val   - mean[None, None, :]) / std[None, None, :]

            # compute class weight (pos_weight = n_neg / n_pos) using TRAIN labels only
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            self.pos_weight = float(n_neg) / max(1.0, float(n_pos))
            print(f" Class balance (train): {n_pos} positives, {n_neg} negatives → pos_weight={self.pos_weight:.2f}")

            # convert to tensors and datasets (event-wise)
            Xt = torch.tensor(X_train, dtype=torch.float32)          # (n_events, N, F)
            coords_t = torch.tensor(coords_train, dtype=torch.float32)  # (n_events, N, 3)
            mt = torch.tensor(mask_train, dtype=torch.bool)
            yt = torch.tensor(y_train, dtype=torch.float32)

            Xv = torch.tensor(X_val, dtype=torch.float32)
            coords_v = torch.tensor(coords_val, dtype=torch.float32)
            mv = torch.tensor(mask_val, dtype=torch.bool)
            yv = torch.tensor(y_val, dtype=torch.float32)

            # Save datasets as TensorDataset of (points, coords, mask, labels)
            self.train_dataset = TensorDataset(Xt, coords_t, mt, yt)
            self.val_dataset   = TensorDataset(Xv, coords_v, mv, yv)

        if stage == "test" or stage is None:
            # Load scaler if available
            scaler_path = os.path.join(self.cfg.training.models_dir, "scaler.npz")
            if os.path.exists(scaler_path):
                scaler = np.load(scaler_path)
                self.mean = scaler["mean"]
                self.std = scaler["std"]
                print(f" Loaded scaler from {scaler_path}")
            else:
                raise RuntimeError(f"Scaler not found at {scaler_path} — cannot normalize test data.")

            test_files = self._get_files(self.cfg.dataset.test_dataset, self.cfg.dataset.test_dir)
            print(f" Found {len(test_files)} test files")
            data_test = ak.from_parquet(test_files)
            X_test, coords_test, mask_test, y_test = build_jet_tensors(data_test, self.features, max_jets=self.max_jets)
            X_test = (X_test - self.mean[None, None, :]) / self.std[None, None, :]
            Xt = torch.tensor(X_test, dtype=torch.float32)
            coords_t = torch.tensor(coords_test, dtype=torch.float32)
            mt = torch.tensor(mask_test, dtype=torch.bool)
            yt = torch.tensor(y_test, dtype=torch.float32)
            self.test_dataset = TensorDataset(Xt, coords_t, mt, yt)

    @staticmethod
    def _collate_to_dict(batch):
        """
        Convert a batch (list of tuples) to dict:
          - uses default_collate to stack per-field
          - returns: {"points","points_xyz","points_mask","labels"}
        """
        # default_collate expects list inputs, so use it directly
        collated = default_collate(batch)  # returns tuple stacked tensors
        # collated is a tuple: (points, coords, mask, labels)
        points, coords, mask, labels = collated
        # mask = mask.unsqueeze(-1) 
        return {
            "points": points,            # (B, N, F)
            "points_xyz": coords,        # (B, N, 4)
            "points_mask": mask,         # (B, N)
            "labels": labels,            # (B, N)
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_to_dict,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_to_dict,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_to_dict,
        )

# def build_jet_tensors(data: ak.Array, features: list, max_jets: int = 14):
#     """
#     Build padded (n_events, max_jets, n_features) jet tensors for jet-level training.

#     Returns:
#         X:       float32 (n_events, max_jets, n_features)
#         mask:    bool    (n_events, max_jets)
#         labels:  float32 (n_events, max_jets)  # isVBF flags
#     """

#     jets_raw = data.TrainingJet                         # all jet fields
#     jets_p4  = initialize_p4(data.TrainingJet)          # pt/eta/phi/mass view

#     # Sort indices per event by descending pt
#     order = ak.argsort(jets_p4.pt, ascending=False)

#     jets_raw  = jets_raw[order]
#     jets_p4   = jets_p4[order]

#     # Get a padded numpy array for a field
#     def get_padded(field_src, field):
#         """Extract field from field_src, pad -> fill -> numpy."""
#         arr = getattr(field_src, field)
#         arr = ak.pad_none(arr, max_jets, clip=True)
#         arr = ak.fill_none(arr, 0)
#         return ak.to_numpy(arr)       # (n_events, max_jets)

#     # Build feature tensor
#     feat_list = []
#     for f in features:
#         # features may come from p4 (pt, eta, phi, mass) or raw fields
#         if hasattr(jets_p4, f):
#             arr = get_padded(jets_p4, f)
#         else:
#             arr = get_padded(jets_raw, f)
#         feat_list.append(arr[..., None])   # add as separate channel

#     X = np.concatenate(feat_list, axis=-1).astype(np.float32)

#     # Build mask (jet exists if original pt > 0)
#     pt_orig = initialize_p4(data.TrainingJet).pt
#     pt_orig = pt_orig[order]
#     pt_orig = ak.pad_none(pt_orig, max_jets, clip=True)
#     pt_orig = ak.fill_none(pt_orig, 0)
#     mask = (ak.to_numpy(pt_orig) > 0)

#     # Labels (isVBF per jet)
#     labels = get_padded(jets_raw, "isVBF").astype(np.float32)

#     return X, mask.astype(bool), labels


# class JetDataModule(LightningDataModule):
#     def __init__(self, cfg: DictConfig, features: list = None, max_jets: int = 14):
#         super().__init__()
#         self.cfg = cfg
#         self.features = features or ["pt", "eta", "phi", "mass"]
#         self.max_jets = max_jets
#         self.batch_size = cfg.training.dataloader.batch_size
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
#         # scaler
#         self.mean = None
#         self.std = None
#         self.pos_weight = 1.0

#     def _get_files(self, dataset_keys, split_dir):
#         dataset_paths = []
#         for key in dataset_keys:
#             base = self.cfg.dataset.datasets[key]
#             path = os.path.join(base, split_dir)
#             files = glob.glob(os.path.join(path, "*.parquet"))
#             dataset_paths.extend(files)
#         return sorted(dataset_paths)

#     def setup(self, stage=None):
#         # train/val
#         if stage == "fit" or stage is None:
#             train_files = self._get_files(self.cfg.dataset.train_dataset, self.cfg.dataset.train_dir)
#             val_files   = self._get_files(self.cfg.dataset.val_dataset, self.cfg.dataset.val_dir)
#             print(f" Found {len(train_files)} train files, {len(val_files)} val files")

#             data_train = ak.from_parquet(train_files)
#             X_train, mask_train, y_train = build_jet_tensors(data_train, self.features, max_jets=self.max_jets)

#             data_val = ak.from_parquet(val_files)
#             X_val, mask_val, y_val = build_jet_tensors(data_val, self.features, max_jets=self.max_jets)

#             # compute scaler from TRAIN only (only real jets)
#             mask_flat = mask_train.reshape(-1)
#             X_flat = X_train.reshape(-1, X_train.shape[-1])
#             valid = mask_flat
#             if valid.sum() == 0:
#                 raise RuntimeError("No valid jets in training data.")
#             mean = X_flat[valid].mean(axis=0)
#             std = X_flat[valid].std(axis=0) + 1e-6
#             self.mean = mean
#             self.std = std
            
#             # Save scaler for later inference
#             scaler_path = os.path.join(self.cfg.training.models_dir, "scaler.npz")
#             os.makedirs(self.cfg.training.models_dir, exist_ok=True)
#             np.savez(scaler_path, mean=mean, std=std)
#             print(f" Saved scaler → {scaler_path}")

#             # apply
#             X_train = (X_train - mean[None, None, :]) / std[None, None, :]
#             X_val   = (X_val   - mean[None, None, :]) / std[None, None, :]

#             # compute class weight (pos_weight = n_neg / n_pos)
#             n_pos = (y_train == 1).sum()
#             n_neg = (y_train == 0).sum()
#             self.pos_weight = float(n_neg) / max(1.0, float(n_pos))
#             print(f" Class balance (train): {n_pos} positives, {n_neg} negatives → pos_weight={self.pos_weight:.2f}")

#             # convert to tensors and datasets (flatten to event batches)
#             # We return event-wise tensors (batch dim in DataLoader will be events)
#             Xt = torch.tensor(X_train, dtype=torch.float32)
#             mt = torch.tensor(mask_train, dtype=torch.bool)
#             yt = torch.tensor(y_train, dtype=torch.float32)

#             Xv = torch.tensor(X_val, dtype=torch.float32)
#             mv = torch.tensor(mask_val, dtype=torch.bool)
#             yv = torch.tensor(y_val, dtype=torch.float32)

#             self.train_dataset = TensorDataset(Xt, mt, yt)
#             self.val_dataset   = TensorDataset(Xv, mv, yv)

#         if stage == "test" or stage is None:
#             # Load scaler if available
#             scaler_path = os.path.join(self.cfg.training.models_dir, "scaler.npz")
#             if os.path.exists(scaler_path):
#                 scaler = np.load(scaler_path)
#                 self.mean = scaler["mean"]
#                 self.std = scaler["std"]
#                 print(f" Loaded scaler from {scaler_path}")
#             else:
#                 raise RuntimeError(f"Scaler not found at {scaler_path} — cannot normalize test data.")

#             test_files = self._get_files(self.cfg.dataset.test_dataset, self.cfg.dataset.test_dir)
#             print(f" Found {len(test_files)} test files")
#             data_test = ak.from_parquet(test_files)
#             X_test, mask_test, y_test = build_jet_tensors(data_test, self.features, max_jets=self.max_jets)
#             X_test = (X_test - self.mean[None, None, :]) / self.std[None, None, :]
#             Xt = torch.tensor(X_test, dtype=torch.float32)
#             mt = torch.tensor(mask_test, dtype=torch.bool)
#             yt = torch.tensor(y_test, dtype=torch.float32)
#             self.test_dataset = TensorDataset(Xt, mt, yt)

            
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)