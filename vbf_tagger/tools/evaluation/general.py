import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import awkward as ak
from omegaconf import DictConfig
from vbf_tagger.tools.visualization import losses as l


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray) or isinstance(obj, ak.Array):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


def filter_losses(metrics_path: str):
    data = pd.read_csv(metrics_path)
    last_epoch = int(max(data["epoch"].dropna()))
    epoch_last_train_losses = []
    all_epoch_train_losses = []
    for idx_epoch in range(last_epoch + 1):
        epoch_train_losses = np.array(data.loc[data["epoch"] == idx_epoch, "train_loss"])
        epoch_train_losses = epoch_train_losses[~np.isnan(epoch_train_losses)]
        all_epoch_train_losses.append(epoch_train_losses)
        epoch_last_train_losses.append(epoch_train_losses[-1])
    val_losses = np.array(data["val_loss"])
    val_losses = val_losses[~np.isnan(val_losses)]
    return {"val_loss": val_losses, "train_loss": epoch_last_train_losses, "all_train_losses": all_epoch_train_losses}


def collect_all_results(predictions_dir: str, cfg: DictConfig, target: str = "target") -> dict:
    """
    Collect all results from the predictions directory.

    Parameters:
        predictions_dir (str): Path to the predictions directory.
        cfg (DictConfig): Configuration.

    Returns:
        dict: Dictionary containing results for each.
    """
    results = {}
    energies = cfg.dataset.particle_energies
    particle_types = cfg.dataset.particle_types
    for pid in particle_types:
        results[pid] = {}
        for energy in energies:
            file_name = f"{energy}_1_pred.parquet" if cfg.dataset.name == "FCC" else f"signal_{pid}_{energy}_*.parquet"
            pid_energy_wcp = os.path.join(predictions_dir, "test", file_name)
            pid_energy_files = list(glob.glob(pid_energy_wcp))
            if len(pid_energy_files) == 0:
                warnings.warn(f"No prediction files found for {pid} at {energy} GeV in {predictions_dir}.")
                continue
            pid_energy_true = []
            pid_energy_pred = []
            for pid_energy_file in pid_energy_files:
                data = ak.from_parquet(pid_energy_file)
                pid_energy_true.append(data[target])
                pid_energy_pred.append(data["pred"])
            pid_energy_pred = ak.concatenate(pid_energy_pred, axis=0)
            pid_energy_true = ak.concatenate(pid_energy_true, axis=0)
            results[pid][energy] = {"true": pid_energy_true, "pred": pid_energy_pred}
    return results


def evaluate_losses(metrics_path: str, model_name: str = "", loss_name: str = "BCE", results_dir: str = ""):
    # Visualize losses for the training.
    losses = filter_losses(metrics_path=metrics_path)
    losses_output_path = os.path.join(results_dir, "losses.png")

    lp = l.LossesMultiPlot(loss_name=loss_name, plot_train_losses=True)
    loss_results = {model_name: losses}
    lp.plot_algorithms(results=loss_results, output_path=losses_output_path)