import os
import glob
import uproot
import random
import numpy as np
import awkward as ak
from torch.utils.data import ConcatDataset, Subset


def load_root_file(path: str, tree_path: str = "sim", branches: list = None) -> ak.Array:
    """Loads the CEPC dataset .root file.

    Parameters:
        path : str
            Path to the .root file
        tree_path : str
            Path in the tree in the .root file.
        branches : list
            [default: None] Branches to be loaded from the .root file. By default, all branches will be loaded.

    Returns:
        array : ak.Array
            Awkward array containing the .root file data
    """
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        arrays = tree.arrays(branches)
    return arrays


def get_all_paths(input_loc, n_files: int = None) -> list:
    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the
    files are located or a wildcard path.

    Parameters:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns
            will be loaded

    Returns:
        input_paths : list
            List of all the .parquet files found in the input location
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_paths = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_paths = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_paths = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_paths = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc: {input_loc}")
    else:
        raise ValueError(f"Unexpected input_loc: {input_loc}")
    return input_paths


def get_row_groups(input_paths: list) -> list:
    """Get the row groups of the input files. The row groups are used to split the data into smaller chunks for
    processing.

    Parameters:
        input_paths : list
            List of all the .parquet files found in the input location

    Returns:
        row_groups : list
            List of all the row groups found in the input files
    """
    row_groups = []
    for data_path in input_paths:
        metadata = ak.metadata_from_parquet(data_path)
        num_row_groups = metadata["num_row_groups"]
        col_counts = metadata["col_counts"]
        row_groups.extend(
            [RowGroup(data_path, row_group, col_counts[row_group]) for row_group in range(num_row_groups)]
        )
    return row_groups


def save_array_to_file(data: ak.Array, output_path: str) -> None:
    print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path, row_group_size=1024)


class RowGroup:
    """Class to represent a row group in a .parquet file. The row group is used to split the data into smaller chunks
    for processing."""

    def __init__(self, filename, row_group, num_rows):
        """Initializes the row group.
        Parameters:
            filename : str
                Name of the .parquet file
            row_group : int
                Row group number
            num_rows : int
                Number of rows in the row group
        """
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


def train_val_split_shuffle(
    concat_dataset: ConcatDataset,
    val_split: float = 0.2,
    seed: int = 42,
    max_waveforms_for_training: int = -1,
    row_group_size: int = 1024,
):
    total_len = len(concat_dataset)
    indices = list(range(total_len))
    random.seed(seed)
    random.shuffle(indices)

    split = int(total_len * val_split)
    if max_waveforms_for_training == -1:
        train_end_idx = None
    else:
        num_train_rows = int(np.ceil(max_waveforms_for_training / row_group_size))
        train_end_idx = split + num_train_rows
    val_indices = indices[:split]
    train_indices = indices[split:train_end_idx]

    train_subset = Subset(concat_dataset, train_indices)
    val_subset = Subset(concat_dataset, val_indices)

    return train_subset, val_subset