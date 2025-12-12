# vbf_tagger/tools/data/flatpair_dataloader.py
import os
import glob
import math
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from omegaconf import OmegaConf
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from vbf_tagger.tools.data import io
from vbf_tagger.tools.data.dataloaders import BaseDataModule
from vbf_tagger.tools.data.general import initialize_p4

def build_flat_pair_features(data: ak.Array):
    # Select events with > 3 jets
    jets = initialize_p4(data.TrainingJet)
    mask_valid = ak.num(jets) > 3
    jets = jets[mask_valid]
    isVBF = data.TrainingJet.isVBF[mask_valid]
    jets_novec = data.TrainingJet[mask_valid]

    # Build all jet pairs
    pairs = ak.combinations(jets, 2, fields=["j1", "j2"])
    pairs_isVBF = ak.combinations(isVBF, 2, fields=["j1", "j2"])
    pairs_novec = ak.combinations(jets_novec, 2, fields=["j1", "j2"])

    # Compute pair features
    mjj = ak.flatten((pairs.j1 + pairs.j2).mass)
    deta = ak.flatten(abs(pairs.j1.eta - pairs.j2.eta))
    dphi = ak.flatten(abs(pairs.j1.phi - pairs.j2.phi))
    ptjj = ak.flatten((pairs.j1 + pairs.j2).pt)
    dRjj = ak.flatten(pairs.j1.deltaR(pairs.j2))
    etaetajj = ak.flatten(pairs.j1.eta * pairs.j2.eta)
    denergyjj = ak.flatten(abs(pairs.j1.energy - pairs.j2.energy))
    ejj = ak.flatten((pairs.j1 + pairs.j2).energy)
    e_mjj = ejj / mjj
    higher_pt_mask = abs(pairs.j1.pt) > abs(pairs.j2.pt)
    min_pt_pair = ak.flatten(ak.where(higher_pt_mask, pairs.j2.pt, pairs.j1.pt))

    # b-tag sums
    btagDeepFlavB_sum = ak.flatten((pairs_novec.j1.btagDeepFlavB + pairs_novec.j2.btagDeepFlavB))
    btagDeepFlavCvB_sum = ak.flatten((pairs_novec.j1.btagDeepFlavCvB + pairs_novec.j2.btagDeepFlavCvB))
    btagDeepFlavCvL_sum = ak.flatten((pairs_novec.j1.btagDeepFlavCvL + pairs_novec.j2.btagDeepFlavCvL))
    btagDeepFlavQG_sum = ak.flatten((pairs_novec.j1.btagDeepFlavQG + pairs_novec.j2.btagDeepFlavQG))
    btagPNetB_sum = ak.flatten((pairs_novec.j1.btagPNetB + pairs_novec.j2.btagPNetB))
    btagPNetCvB_sum = ak.flatten((pairs_novec.j1.btagPNetCvB + pairs_novec.j2.btagPNetCvB))
    btagPNetCvL_sum = ak.flatten((pairs_novec.j1.btagPNetCvL + pairs_novec.j2.btagPNetCvL))
    btagPNetCvNotB_sum = ak.flatten((pairs_novec.j1.btagPNetCvNotB + pairs_novec.j2.btagPNetCvNotB))
    btagPNetQvG_sum = ak.flatten((pairs_novec.j1.btagPNetQvG + pairs_novec.j2.btagPNetQvG))
    btagPNetTauVJet_sum = ak.flatten((pairs_novec.j1.btagPNetTauVJet + pairs_novec.j2.btagPNetTauVJet))
    hhbtag_sum = ak.flatten((pairs_novec.j1.hhbtag + pairs_novec.j2.hhbtag))

    # # MET
    # PuppiMET_covXY = data.PuppiMET.covXY[mask_valid]
    # PuppiMET_pt = data.PuppiMET.pt[mask_valid]
    # PuppiMET_covXY_per_pair, _ = ak.broadcast_arrays(PuppiMET_covXY, pairs)
    # PuppiMET_pt_per_pair, _ = ak.broadcast_arrays(PuppiMET_pt, pairs)

    # event-level vars
    event_energy = ak.sum(jets.energy, axis=1)
    event_energy_per_pair, _ = ak.broadcast_arrays(event_energy, pairs)
    eventenergy = ak.flatten(event_energy_per_pair)
    # event_pt = ak.sum(jets.pt, axis=1)
    # event_pt_per_pair, _ = ak.broadcast_arrays(event_pt, pairs)

    targets = ak.flatten((pairs_isVBF.j1 == 1) & (pairs_isVBF.j2 == 1))

    features = ak.Array({
        "mjj": mjj,
        # "ptjj": ptjj,
        # "deta": deta,
        # "dphi": dphi,
        # "btagDeepFlavB_sum": btagDeepFlavB_sum,
        # "btagDeepFlavCvB_sum": btagDeepFlavCvB_sum,
        # "btagDeepFlavCvL_sum": btagDeepFlavCvL_sum,
        # "btagDeepFlavQG_sum": btagDeepFlavQG_sum,
        "btagPNetB_sum": btagPNetB_sum,
        # "btagPNetCvB_sum": btagPNetCvB_sum,
        # "btagPNetCvL_sum": btagPNetCvL_sum,
        # "btagPNetCvNotB_sum": btagPNetCvNotB_sum,
        # "btagPNetQvG_sum": btagPNetQvG_sum,
        # "btagPNetTauVJet_sum": btagPNetTauVJet_sum,
        # "hhbtag_sum": hhbtag_sum,
        # "e_mjj": e_mjj,
        # "dRjj": dRjj,
        "etaetajj": etaetajj,
        "denergyjj": denergyjj,
        # "min_pt_pair": min_pt_pair,
        # "PuppiMET_covXY_per_pair": PuppiMET_covXY_per_pair,
        # "PuppiMET_pt_per_pair": PuppiMET_pt_per_pair,   
        # "event_energy_per_pair": event_energy_per_pair,
        # "event_pt_per_pair": event_pt_per_pair,
        "eventenergy": eventenergy,
    })

    X = ak.to_numpy(ak.values_astype(features, np.float32))
    X = np.stack([X[name] for name in X.dtype.names], axis=1)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    y = ak.to_numpy(ak.values_astype(targets, np.int8))
    y = np.array(y)

    return X, y


class FlatPairDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_weight = None
        self.batch_size = cfg.training.dataloader.batch_size

    def _get_files(self, dataset_keys, split_dir):
        """Collect all parquet files for a given dataset"""
        dataset_paths = []
        for key in dataset_keys:
            base = self.cfg.dataset.datasets[key]
            path = os.path.join(base, split_dir)
            files = glob.glob(os.path.join(path, "*.parquet"))
            dataset_paths.extend(files)
        return dataset_paths

    def setup(self, stage=None):
        # Load train/val files
        if stage == "fit" or stage is None:
            train_files = self._get_files(self.cfg.dataset.train_dataset, self.cfg.dataset.train_dir)
            val_files   = self._get_files(self.cfg.dataset.val_dataset,   self.cfg.dataset.val_dir)

            print(f" Found {len(train_files)} train files, {len(val_files)} val files")

            # Build flat pair features
            train_data = ak.from_parquet(train_files)
            X_train, y_train = build_flat_pair_features(train_data)

            val_data = ak.from_parquet(val_files)
            X_val, y_val = build_flat_pair_features(val_data)

            # Compute class weight
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            self.pos_weight = n_neg / n_pos
            print(f" Class balance (train): {n_pos} positives, {n_neg} negatives → pos_weight={self.pos_weight:.2f}")

            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val   = torch.tensor(X_val,   dtype=torch.float32)
            y_val   = torch.tensor(y_val,   dtype=torch.float32)

            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset   = TensorDataset(X_val, y_val)
        
        # Load test files
        if stage == "test" or stage is None:
            test_files = self._get_files(self.cfg.dataset.test_dataset, self.cfg.dataset.test_dir)
            print(f" Found {len(test_files)} test files")

            test_data = ak.from_parquet(test_files)
            X_test, y_test = build_flat_pair_features(test_data)

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
