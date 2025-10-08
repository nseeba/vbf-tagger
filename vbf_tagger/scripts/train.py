import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*onnxscript\.values\.OnnxFunction\.param_schemas.*",
    category=FutureWarning,
    module=r"onnxscript\.converter",
)

import os
import glob
import hydra
from hydra.utils import instantiate
import torch
import lightning as L
import awkward as ak
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from vbf_tagger.tools.data import dataloaders as dl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from vbf_tagger.models.LorentzNet import classification


torch.set_float32_matmul_precision("medium")  # or 'high'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FlatCSVLogger(CSVLogger):
    def __init__(self, save_dir, name):
        # No name or version
        super().__init__(save_dir=save_dir, name=name, version="")

    @property
    def log_dir(self):
        # Skip versioned subdirectory
        return self.save_dir


def base_train(cfg: DictConfig, models_dir: str):
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=models_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        # save_weights=True,
        filename="model_best",
    )
    # early_stop = EarlyStopping(monitor="val_loss", patience=6, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    max_epochs = 2 if cfg.training.debug_run else cfg.training.trainer.max_epochs
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback_best, lr_monitor],
        logger=FlatCSVLogger(save_dir=cfg.training.log_dir, name="metrics"),
        overfit_batches=1 if cfg.training.debug_run else 0,
        num_sanity_val_steps=0,
    )
    # trainer = L.Trainer(
    #     max_epochs=max_epochs,
    #     callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback_best, early_stop, lr_monitor],
    #     logger=FlatCSVLogger(save_dir=cfg.training.log_dir, name="metrics"),
    #     overfit_batches=1 if cfg.training.debug_run else 0,
    #     num_sanity_val_steps=0,
    # )
    return trainer, checkpoint_callback_best

# def train_vbf(cfg: DictConfig, data_type: str):
#     print(f"Training {cfg.models.classification.model.name} for VBF classification.")
#     model = instantiate(cfg.models.classification.model)
#     models_dir = cfg.training.models_dir

#     # Use VBFDataModule
#     datamodule = dl.VBFDataModule(
#         cfg=cfg,
#         data_type=data_type,
#         debug_run=cfg.training.debug_run,
#         device=DEVICE,
#     )

#     metrics_path = os.path.join(cfg.training.log_dir, "metrics.csv")
#     best_model_path = os.path.join(cfg.training.models_dir, "model_best.ckpt")

#     if not cfg.training.model_evaluation:
#         trainer, checkpoint_callback = base_train(cfg, models_dir=models_dir)
#         trainer.fit(model=model, datamodule=datamodule)
#         best_model_path = checkpoint_callback.best_model_path
#         metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
#     else:
#         # If just evaluating, load pre-trained checkpoint and metrics
#         if not os.path.exists(metrics_path) or not os.path.exists(best_model_path):
#             best_model_path = cfg.models.classification.model.checkpoint.model
#             metrics_path = cfg.models.classification.model.checkpoint.losses

#     return model, best_model_path, metrics_path


def train_vbf(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.classification.model.name} for VBF classification.")
    models_dir = cfg.training.models_dir

    # Build datamodule first
    datamodule = dl.VBFDataModule(
        cfg=cfg,
        data_type=data_type,
        debug_run=cfg.training.debug_run,
        device=DEVICE,
    )
    datamodule.setup(stage="fit")  # force setup so pos_weight is computed

    # Now build model, injecting pos_weight
    model = classification(
        name="classification",
        hyperparameters={**cfg.models.classification.model.hyperparameters,
                         "pos_weight": getattr(datamodule, "pos_weight", 1.0)},
    )

    metrics_path = os.path.join(cfg.training.log_dir, "metrics.csv")
    best_model_path = os.path.join(cfg.training.models_dir, "model_best.ckpt")

    if not cfg.training.model_evaluation:
        trainer, checkpoint_callback = base_train(cfg, models_dir=models_dir)
        trainer.fit(model=model, datamodule=datamodule)
        best_model_path = checkpoint_callback.best_model_path
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        if not os.path.exists(metrics_path) or not os.path.exists(best_model_path):
            best_model_path = cfg.models.classification.model.checkpoint.model
            metrics_path = cfg.models.classification.model.checkpoint.losses

    return model, best_model_path, metrics_path



def train_one_step(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.one_step.model.name} for the one-step training.")
    model = instantiate(cfg.models.one_step.model)
    # datamodule = dl.OneStepDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    models_dir = cfg.training.models_dir
    datamodule = dl.OneStepWindowedDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    metrics_path = os.path.join(cfg.training.log_dir, "metrics.csv")
    best_model_path = os.path.join(cfg.training.models_dir, "model_best.ckpt")
    if not cfg.training.model_evaluation:
        trainer, checkpoint_callback = base_train(cfg, models_dir=models_dir)
        trainer.fit(model=model, datamodule=datamodule)
        best_model_path = checkpoint_callback.best_model_path
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        if not os.path.exists(metrics_path) or not os.path.exists(best_model_path):
            best_model_path = cfg.models.one_step.model.checkpoint.model
            metrics_path = cfg.models.one_step.model.checkpoint.losses
    return model, best_model_path, metrics_path

def save_predictions(input_path: str, all_predictions: ak.Array, all_targets: ak.Array, cfg: DictConfig, scenario: str):
    additional_dirs = ["two_step_pf", "two_step_cl"]
    if not scenario == "two_step_cl":
        predictions_dir = cfg.training.predictions_dir
        print("prediction_dir:", predictions_dir)
        base_scenario = (
            "two_step" if "two_step" in scenario else "two_step"
        )  # Temporary, as atm also one-step-windowed uses two-step ntuples
        additional_dir_level = scenario if scenario in additional_dirs else ""
        base_dir = cfg.dataset.data_dir
        original_dir = os.path.join(base_dir, base_scenario)
        predictions_dir = os.path.join(predictions_dir, additional_dir_level)
        os.makedirs(predictions_dir, exist_ok=True)
        output_path = input_path.replace(original_dir, predictions_dir)
        output_path = output_path.replace(".parquet", "_pred.parquet")
    else:
        output_path = input_path.replace("two_step_pf", "two_step_cl")

    input_data = ak.from_parquet(input_path)
    output_data = ak.copy(input_data)
    output_data["pred"] = ak.Array(all_predictions)  # pylint: disable=E1137
    output_data["pad_targets"] = ak.Array(all_targets)  # pylint: disable=E1137
    print(f"Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ak.to_parquet(output_data, output_path, row_group_size=cfg.preprocessing.row_group_size)


def create_prediction_files(file_list: list, iterable_dataset: IterableDataset, model, cfg: DictConfig, scenario: str):
    num_files = 2 if cfg.training.debug_run else None
    print("Creating prediction files")
    print("file list", file_list)
    with torch.no_grad():
        for path in file_list[:num_files]:
            print("Processing path: ", path)
            dataset = dl.RowGroupDataset(path)
            iterable_dataset_ = iterable_dataset(dataset, device=DEVICE, cfg=cfg)
            dataloader = DataLoader(
                dataset=iterable_dataset_,
                batch_size=cfg.training.dataloader.batch_size,
                # num_workers=cfg.training.dataloader.num_dataloader_workers,
                # prefetch_factor=cfg.training.dataloader.prefetch_factor,
            )
            all_predictions = []
            all_targets = []
            for i, batch in enumerate(dataloader):
                predictions, targets = model(batch)

                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
            all_predictions = ak.concatenate(all_predictions, axis=0)
            all_targets = ak.concatenate(all_targets, axis=0)
            save_predictions(
                input_path=path, all_predictions=all_predictions, all_targets=all_targets, cfg=cfg, scenario=scenario
            )


def evaluate_vbf(cfg: DictConfig, model, metrics_path: str):
    print("Evaluating VBF model...")
    datamodule = dl.VBFDataModule(
        cfg=cfg,
        data_type="",  # same as train
        debug_run=cfg.training.debug_run,
        device=DEVICE,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=False,
    )
    results = trainer.validate(model, datamodule=datamodule)
    print("Validation results:", results)
    return results



def evaluate_one_step(cfg: DictConfig, model, metrics_path: str) -> list:
    model.to(DEVICE)
    model.eval()
    dir_ = "*" if cfg.evaluation.training.eval_all else "test"
    wcp_path = os.path.join(cfg.dataset.data_dir, "two_step", dir_, "*")
    file_list = glob.glob(wcp_path)
    # iterable_dataset = dl.OneStepIterableDataset
    iterable_dataset = dl.OneStepWindowedIterableDataset

    # Create prediction files
    # create_prediction_files(file_list, iterable_dataset=iterable_dataset, model=model, cfg=cfg, scenario="one_step")

    # Evaluate training
    ose.evaluate_training(cfg=cfg, metrics_path=metrics_path)


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    if cfg.training.debug_run:
        print("Running in debug mode, only a few epochs and files will be processed.")
    training_type = cfg.training.type
    if training_type == "classification":
        print("Training classification model.")
        model, best_model_path, metrics_path = train_vbf(cfg, data_type="")
        if cfg.training.model_evaluation:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            evaluate_vbf(cfg, model, metrics_path)
    else:
        raise ValueError(f"Unknown training type: {training_type}")


if __name__ == "__main__":
    # mp.set_start_method("fork")
    mp.set_start_method('spawn', force=True)
    main()  # pylint: disable=E1120