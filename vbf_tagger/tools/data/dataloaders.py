import os
import math
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from omegaconf import OmegaConf
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset
from vbf_tagger.tools.data import io
from vbf_tagger.tools.data.general import initialize_p4, stack_and_pad_features



##########################################################################
##########################################################################
######################             Base classes            ###############
##########################################################################
##########################################################################


class RowGroupDataset(Dataset):
    def __init__(self, data_loc: str):
        self.data_loc = data_loc
        self.input_paths = io.get_all_paths(data_loc)
        self.row_groups = io.get_row_groups(self.input_paths)

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class BaseIterableDataset(IterableDataset):
    """Base iterable dataset class to be used for different types of trainings."""

    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.row_groups = [row_group for row_group in self.dataset]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        self.device = device
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms/events/jets in the dataset.")

    def build_tensors(self, data: ak.Array):
        """Builds the input and target tensors from the data.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        raise NotImplementedError("Please implement the build_tensors method in your subclass")

    # def __len__(self):
    #     return self.num_rows

    def _move_to_device(self, batch):
        if isinstance(batch, (tuple, list)):
            return [self._move_to_device(x) for x in batch]
        return batch.to(self.device)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            row_groups_start = worker_id * per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]
        for row_group in row_groups_to_process:
            # load one chunk from one file
            data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
            tensors = self.build_tensors(data)
            # return individual jets from the dataset
            for idx_wf in range(len(tensors[0])):
            # for idx_wf in range(len(data)):
                # features, targets
                yield (
                    self._move_to_device(tensors[0][idx_wf]),
                    self._move_to_device(tensors[1][idx_wf]),
                    self._move_to_device(tensors[2][idx_wf]),
                    self._move_to_device(tensors[3][idx_wf]),
                )


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        iter_dataset: IterableDataset,
        data_type: str,
        debug_run: bool = False,
        device: str = "cpu",
        clusterization: bool = False,
    ):
        """Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.
            iter_dataset : IterableDataset
                The iterable dataset to be used for training and validation.
                Need to define a separate class for each training type, e.g. one_step, two_step_peak_finding,
                two_step_clusterization, two_step_minimal etc.
            data_type : str
                The type of the data. In case of CEPC it can be "kaon" or "pion".
                In case of FCC it is the different energies.
        """
        self.cfg = cfg
        # self.task = "two_step" if self.cfg.training.type == "two_step_minimal" else self.cfg.training.type
        self.task = "classification"
        # self.task = "two_step"
        self.debug_run = debug_run
        self.data_type = data_type
        self.iter_dataset = iter_dataset
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.clusterization = clusterization
        self.num_row_groups = 2 if debug_run else None
        self.save_hyperparameters()
        super().__init__()

    def setup(self, stage: str) -> None:
        batch_size = self.cfg.training.dataloader.batch_size if not self.debug_run else 1

        def make_dataset(subsets):
            datasets = []
            for name in subsets:
                root = self.cfg.dataset.datasets[name]  # e.g. vbf, ggf, tt_sl paths
                split_dir = os.path.join(root, self.cfg.dataset.train_dir if stage=="fit" else self.cfg.dataset.test_dir)
                ds = RowGroupDataset(data_loc=split_dir)[: self.num_row_groups]
                datasets.append(ds)
            return datasets

        if stage == "fit":
            train_datasets = make_dataset(self.cfg.dataset.train_dataset)
            val_datasets   = make_dataset(self.cfg.dataset.val_dataset)

            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.val_dataset   = torch.utils.data.ConcatDataset(val_datasets)

            self.train_dataset = self.iter_dataset(dataset=self.train_dataset, device=self.device, cfg=self.cfg)
            self.val_dataset   = self.iter_dataset(dataset=self.val_dataset, device=self.device, cfg=self.cfg)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
            )

        elif stage == "test":
            test_datasets = make_dataset(self.cfg.dataset.test_dataset)
            self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)
            self.test_dataset = self.iter_dataset(dataset=self.test_dataset, device=self.device, cfg=self.cfg)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")


    # def setup(self, stage: str) -> None:
    #     batch_size = self.cfg.training.dataloader.batch_size if not self.debug_run else 1
    #     if stage == "fit":
    #         train_dir = os.path.join(self.cfg.dataset.data_dir, self.cfg.dataset.train_dir)
    #         val_dir   = os.path.join(self.cfg.dataset.data_dir, self.cfg.dataset.val_dir)
    #         self.train_dataset = RowGroupDataset(data_loc=train_dir)[: self.num_row_groups]
    #         self.val_dataset   = RowGroupDataset(data_loc=val_dir)[: self.num_row_groups]
    #         self.train_dataset = self.iter_dataset(dataset=self.train_dataset, device=self.device, cfg=self.cfg)
    #         self.val_dataset = self.iter_dataset(dataset=self.val_dataset, device=self.device, cfg=self.cfg)
    #         self.train_loader = DataLoader(
    #             self.train_dataset,
    #             batch_size=batch_size,
    #             persistent_workers=True,
    #             num_workers=self.cfg.training.dataloader.num_dataloader_workers,
    #             # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
    #         )
    #         self.val_loader = DataLoader(
    #             self.val_dataset,
    #             batch_size=batch_size,
    #             persistent_workers=True,
    #             num_workers=self.cfg.training.dataloader.num_dataloader_workers,
    #             # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
    #         )
    #     elif stage == "test":
    #         test_dir = os.path.join(self.cfg.dataset.data_dir, self.cfg.dataset.test_dir)
    #         self.test_dataset = RowGroupDataset(data_loc=test_dir)[: self.num_row_groups]
    #         self.test_dataset = self.iter_dataset(dataset=self.test_dataset, device=self.device, cfg=self.cfg)
    #         self.test_loader = DataLoader(
    #             self.test_dataset,
    #             batch_size=batch_size,
    #             persistent_workers=True,
    #             num_workers=self.cfg.training.dataloader.num_dataloader_workers,
    #             # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
    #         )
    #     else:
    #         raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

class VBFIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)
        self.max_pairs = cfg.dataset.get("max_pairs", 30)  # choose a cutoff (e.g. 45 = C(10,2))
    #     self._length = 0
    #     for row_group in self.row_groups:
    #         data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
    #         jets = initialize_p4(data.TrainingJet)
    #         mask_valid = ak.num(jets) > 3
    #         num_pairs = ak.num(ak.combinations(jets[mask_valid], 2))
    #         self._length += int(ak.sum(np.minimum(num_pairs, self.max_pairs)))

    # def __len__(self):
    #     return self._length


    def build_tensors(self, data: ak.Array):
        # jet 4-vectors (for kinematics)
        jets = initialize_p4(data.TrainingJet)

        # keep only events with > 3 jets
        mask_valid = ak.num(jets) > 3
        jets = jets[mask_valid]
        isVBF = data.TrainingJet.isVBF[mask_valid]

        # jet objects
        jets_novec = data.TrainingJet[mask_valid]

        # build all jet pairs
        pairs = ak.combinations(jets, 2, fields=["j1", "j2"])
        pairs_isVBF = ak.combinations(isVBF, 2, fields=["j1", "j2"])
        pairs_novec = ak.combinations(jets_novec, 2, fields=["j1", "j2"])

        # kinematics
        pair_p4 = pairs.j1 + pairs.j2 
        cand_kinematics = ak.Array({
            "cand_px": pair_p4.px,
            "cand_py": pair_p4.py,
            "cand_pz": pair_p4.pz,
            "cand_en": pair_p4.energy,
        })
        
        cand_kinematics = ak.pad_none(cand_kinematics, self.max_pairs, clip=True)
        cand_kinematics = ak.fill_none(cand_kinematics, 0)
        
        cand_kinematics_tensors = stack_and_pad_features(cand_kinematics, self.max_pairs)
        cand_kinematics_tensors = torch.tensor(cand_kinematics_tensors, dtype=torch.float32)

        # compute pair-level features
        mjj = (pairs.j1 + pairs.j2).mass
        deta = abs(pairs.j1.eta - pairs.j2.eta)
        dphi = abs(pairs.j1.phi - pairs.j2.phi)
        ptjj = (pairs.j1 + pairs.j2).pt
        dRjj = pairs.j1.deltaR(pairs.j2)
        etaetajj = pairs.j1.eta * pairs.j2.eta
        # etaetajj = abs(pairs.j1.eta * pairs.j2.eta)
        denergyjj = abs(pairs.j1.energy - pairs.j2.energy)
        ejj = (pairs.j1 + pairs.j2).energy
        e_mjj = ejj/mjj
        event_energy = ak.sum(jets.energy, axis=1)
        event_energy_per_pair, _ = ak.broadcast_arrays(event_energy, pairs)
        event_pt = ak.sum(jets.pt, axis=1)
        event_pt_per_pair, _ = ak.broadcast_arrays(event_pt, pairs)
        higher_pt_mask = abs(pairs.j1.pt) > abs(pairs.j2.pt)
        max_pt_pair = ak.where(higher_pt_mask, pairs.j1.pt, pairs.j2.pt)
        min_pt_pair = ak.where(higher_pt_mask, pairs.j2.pt, pairs.j1.pt)
        btagDeepFlavB_sum = (pairs_novec.j1.btagDeepFlavB + pairs_novec.j2.btagDeepFlavB)
        btagDeepFlavCvB_sum = (pairs_novec.j1.btagDeepFlavCvB + pairs_novec.j2.btagDeepFlavCvB)
        btagDeepFlavCvL_sum = (pairs_novec.j1.btagDeepFlavCvL + pairs_novec.j2.btagDeepFlavCvL)
        btagDeepFlavQG_sum = (pairs_novec.j1.btagDeepFlavQG + pairs_novec.j2.btagDeepFlavQG)
        btagPNetB_sum = (pairs_novec.j1.btagPNetB + pairs_novec.j2.btagPNetB)
        btagPNetCvB_sum = (pairs_novec.j1.btagPNetCvB + pairs_novec.j2.btagPNetCvB)
        btagPNetCvL_sum = (pairs_novec.j1.btagPNetCvL + pairs_novec.j2.btagPNetCvL)
        btagPNetCvNotB_sum = (pairs_novec.j1.btagPNetCvNotB + pairs_novec.j2.btagPNetCvNotB)
        btagPNetQvG_sum = (pairs_novec.j1.btagPNetQvG + pairs_novec.j2.btagPNetQvG)
        btagPNetTauVJet_sum = (pairs_novec.j1.btagPNetTauVJet + pairs_novec.j2.btagPNetTauVJet)
        hhbtag_sum = (pairs_novec.j1.hhbtag + pairs_novec.j2.hhbtag)
        PuppiMET_covXY = data.PuppiMET.covXY[mask_valid]
        PuppiMET_pt = data.PuppiMET.pt[mask_valid]
        PuppiMET_covXY_per_pair, _ = ak.broadcast_arrays(PuppiMET_covXY, pairs)
        PuppiMET_pt_per_pair, _ = ak.broadcast_arrays(PuppiMET_pt, pairs)

        # features
        cand_features = ak.Array({
            "mjj": mjj,
            "ptjj": ptjj,
            "deta": deta,
            "dphi": dphi,
            "btagDeepFlavB_sum": btagDeepFlavB_sum,
            "btagDeepFlavCvB_sum": btagDeepFlavCvB_sum,
            "btagDeepFlavCvL_sum": btagDeepFlavCvL_sum,
            "btagDeepFlavQG_sum": btagDeepFlavQG_sum,
            "btagPNetB_sum": btagPNetB_sum,
            "btagPNetCvB_sum": btagPNetCvB_sum,
            "btagPNetCvL_sum": btagPNetCvL_sum,
            "btagPNetCvNotB_sum": btagPNetCvNotB_sum,
            "btagPNetQvG_sum": btagPNetQvG_sum,
            "btagPNetTauVJet_sum": btagPNetTauVJet_sum,
            "hhbtag_sum": hhbtag_sum,
            "PuppiMET_covXY_per_pair": PuppiMET_covXY_per_pair,
            "PuppiMET_pt_per_pair": PuppiMET_pt_per_pair,   
            "event_energy_per_pair": event_energy_per_pair,
            "event_pt_per_pair": event_pt_per_pair,
            "e_mjj": e_mjj,
            "dRjj": dRjj,
            "etaetajj": etaetajj,
            "denergyjj": denergyjj,
            "max_pt_pair": max_pt_pair,
            "min_pt_pair": min_pt_pair,
           
        })

        # pad/truncate to max_pairs
        cand_features = ak.pad_none(cand_features, self.max_pairs, clip=True)
        cand_features = ak.fill_none(cand_features, 0)
        cand_features_tensors = stack_and_pad_features(cand_features, self.max_pairs)
        cand_features_tensors = torch.tensor(cand_features_tensors, dtype=torch.float32)

        targets = (pairs_isVBF.j1==1) & (pairs_isVBF.j2==1)
        targets = ak.pad_none(targets, self.max_pairs, clip=True)
        # mask = which pairs are real (before padding)
        mask_tensor = torch.tensor(ak.to_numpy(~ak.is_none(targets, axis=-1)), dtype=torch.bool).unsqueeze(1)
        targets = ak.fill_none(targets, 0)
        targets_tensor = torch.tensor(ak.to_numpy(targets), dtype=torch.float32)

        return cand_features_tensors, cand_kinematics_tensors, mask_tensor, targets_tensor

class VBFDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for VBF jet tagging."""
        iter_dataset = VBFIterableDataset
        super().__init__(
            cfg=cfg,
            iter_dataset=iter_dataset,
            data_type=data_type,
            debug_run=debug_run,
            device=device,
        )


class VBFDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for VBF jet tagging."""
        iter_dataset = VBFIterableDataset
        super().__init__(
            cfg=cfg,
            iter_dataset=iter_dataset,
            data_type=data_type,
            debug_run=debug_run,
            device=device,
        )

    def setup(self, stage=None):
        # First call parent setup to build datasets + loaders
        super().setup(stage=stage)

        if stage == "fit":
            pos, neg = 0, 0
            for batch in self.train_dataloader():
                _, _, mask, targets = batch
                mask = mask.view(-1).bool()
                t = targets.view(-1)[mask]
                pos += (t == 1).sum().item()
                neg += (t == 0).sum().item()

            pos_weight = neg / pos if pos > 0 else 1.0
            print(f"[INFO] Computed pos_weight = {pos_weight:.3f} from training data")
            # Save into config so the model sees it
            self.pos_weight = pos_weight





# class VBFIterableDataset(BaseIterableDataset):
#     def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
#         super().__init__(dataset, device=device, cfg=cfg)
#         self.max_jets = 16

#     def build_tensors(self, data: ak.Array):
#         """Builds the jet tensors for VBF tagging."""
#         # Build jet four-vectors
#         jet_p4s = initialize_p4(data)

#         cand_kinematics = ak.Array({
#             "cand_px": jet_p4s.px,
#             "cand_py": jet_p4s.py,
#             "cand_pz": jet_p4s.pz,
#             "cand_en": jet_p4s.energy,
#         })

#         cand_kinematics = ak.pad_none(cand_kinematics, self.max_jets, clip=True)
#         cand_kinematics = ak.fill_none(cand_kinematics, 0)

#         cand_kinematics_tensors = stack_and_pad_features(cand_kinematics, self.max_jets)
#         cand_kinematics_tensors = torch.tensor(cand_kinematics_tensors, dtype=torch.float32)
#         # cand_kinematics_tensors = stack_and_pad_features(cand_kinematics, self.max_jets)
#         # cand_kinematics_tensors = torch.tensor(cand_kinematics_tensors, dtype=torch.float32)

#         cand_features = ak.Array({
#                 "btagDeepFlavB": data.btagDeepFlavB,
#                 "btagDeepFlavCvB": data.btagDeepFlavCvB,
#                 "btagDeepFlavCvL": data.btagDeepFlavCvL,
#                 "btagDeepFlavQG": data.btagDeepFlavQG,
#                 "btagPNetB": data.btagPNetB,
#                 "btagPNetCvB": data.btagPNetCvB,
#                 "btagPNetCvL": data.btagPNetCvL,
#                 "btagPNetCvNotB": data.btagPNetCvNotB,
#                 "btagPNetQvG": data.btagPNetQvG,
#                 "btagPNetTauVJet": data.btagPNetTauVJet,
#                 "hhbtag": data.hhbtag,
#             })

#         # cand_features_tensors = stack_and_pad_features(cand_features, self.max_jets)
#         # cand_features_tensors = torch.tensor(cand_features_tensors, dtype=torch.float32)
#         cand_features = ak.pad_none(cand_features, self.max_jets, clip=True)
#         cand_features = ak.fill_none(cand_features, 0)

#         cand_features_tensors = stack_and_pad_features(cand_features, self.max_jets)
#         cand_features_tensors = torch.tensor(cand_features_tensors, dtype=torch.float32)

#         node_mask_tensors = torch.unsqueeze(
#             torch.tensor(
#                 ak.to_numpy(ak.fill_none(ak.pad_none(ak.ones_like(data.isVBF), 16, clip=True), 0,)),
#                 dtype=torch.bool
#             ),
#             dim=1
#         )
#         targets = torch.tensor(
#             ak.to_numpy(ak.fill_none(ak.pad_none(data.isVBF, self.max_jets, clip=True), 0,)),
#             dtype=torch.float32
#         )

#         # print("Returning tensors with shapes:",
#         #     cand_features_tensors.shape,
#         #     cand_kinematics_tensors.shape,
#         #     node_mask_tensors.shape,
#         #     targets.shape)

#         return cand_features_tensors, cand_kinematics_tensors, node_mask_tensors, targets