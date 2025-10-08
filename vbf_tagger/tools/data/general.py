import os
import glob
import json
import vector
import numpy as np
import awkward as ak


# def load_all_data(input_loc: str | list, n_files: int = None, columns: list = None) -> ak.Array:
def load_all_data(input_loc, n_files: int = None, columns: list = None) -> ak.Array:

    """Loads all .parquet files specified by the input. The input can be a list of input_paths, a directory where the files
    are located or a wildcard path.

    Args:
        input_loc : str
            Location of the .parquet files.
        n_files : int
            [default: None] Maximum number of input files to be loaded. By default all will be loaded.
        columns : list
            [default: None] Names of the columns/branches to be loaded from the .parquet file. By default all columns will
            be loaded

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    if n_files == -1:
        n_files = None
    if isinstance(input_loc, list):
        input_files = input_loc[:n_files]
    elif isinstance(input_loc, str):
        if os.path.isdir(input_loc):
            input_loc = os.path.expandvars(input_loc)
            input_files = glob.glob(os.path.join(input_loc, "*.parquet"))[:n_files]
        elif "*" in input_loc:
            input_files = glob.glob(input_loc)[:n_files]
        elif os.path.isfile(input_loc):
            input_files = [input_loc]
        else:
            raise ValueError(f"Unexpected input_loc")
    else:
        raise ValueError(f"Unexpected input_loc")
    input_data = []
    # for file_path in tqdm.tqdm(sorted(input_files)):
    for i, file_path in enumerate(sorted(input_files)):
        print(f"[{i+1}/{len(input_files)}] Loading from {file_path}")
        try:
            input_data.append(load_parquet(file_path, columns=columns))
        except ValueError:
            print(f"{file_path} does not exist")
    if len(input_data) > 0:
        data = ak.concatenate(input_data)
        print("Input data loaded")
    else:
        raise ValueError(f"No files found in {input_loc}")
    return data


def load_parquet(input_path: str, columns: list = None) -> ak.Array:
    """ Loads the contents of the .parquet file specified by the input_path

    Args:
        input_path : str
            The path to the .parquet file to be loaded.
        columns : list
            Names of the columns/branches to be loaded from the .parquet file

    Returns:
        input_data : ak.Array
            The data from the .parquet file
    """
    ret = ak.from_parquet(input_path, columns=columns)
    ret = ak.Array({k: ret[k] for k in ret.fields})
    return ret

def prepare_one_hot_encoding(values, classes=[0, 1, 2, 10, 11, 15]):
    mapping = {class_: i for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)


def one_hot_decoding(values, classes=[0, 1, 2, 10, 11, 15]):
    mapping = {i: class_ for i, class_ in enumerate(classes)}
    return np.vectorize(mapping.get)(values)


def reinitialize_p4(p4_obj: ak.Array):
    """ Reinitialized the 4-momentum for particle in order to access its properties.

    Args:
        p4_obj : ak.Array
            The particle represented by its 4-momenta

    Returns:
        p4 : ak.Array
            Particle with initialized 4-momenta.
    """
    if "tau" in p4_obj.fields:
        p4 = vector.awk(
            ak.zip(
                {
                    "mass": p4_obj.tau,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    else:
        p4 = vector.awk(
            ak.zip(
                {
                    "energy": p4_obj.t,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    return p4

def initialize_p4(data):
    return vector.awk(
            ak.zip(
                {
                    "mass": data.mass,
                    "eta": data.eta,
                    "phi": data.phi,
                    "pt": data.pt,
                }
            )
            )

def stack_and_pad_features(cand_features, max_cands):
    cand_features_tensors = np.stack([ak.pad_none(cand_features[feat], max_cands, clip=True) for feat in cand_features.fields], axis=-1)
    cand_features_tensors = ak.to_numpy(ak.fill_none(cand_features_tensors, 0))
    # Swapping the axes such that it has the shape of (nJets, nFeatures, nParticles)
    cand_features_tensors = np.swapaxes(cand_features_tensors, 1, 2)

    cand_features_tensors[np.isnan(cand_features_tensors)] = 0
    cand_features_tensors[np.isinf(cand_features_tensors)] = 0
    return cand_features_tensors

def load_json(path):
    """ Loads the contents of the .json file with the given path

    Args:
        path : str
            The location of the .json file

    Returns:
        data : dict
            The content of the loaded json file.
    """
    with open(path, "rt") as in_file:
        data = json.load(in_file)
    return data


class NpEncoder(json.JSONEncoder):
    """ Class for encoding various objects such that they could be saved to a json file"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_to_json(data, output_path):
    """ Saves data to a .json file located at `output_path`

    Args:
        data : dict
            The data to be saved
        output_path : str
            Destonation of the .json file

    Returns:
        None
    """
    with open(output_path, "wt") as out_file:
        json.dump(data, out_file, indent=4, cls=NpEncoder)