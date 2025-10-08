import json
from omegaconf import DictConfig, OmegaConf


def print_config(cfg: DictConfig) -> None:
    """Prints the configuration used for the processing

    Parameters:
        cfg : DictConfig
            The configuration to be used

    Returns:
        None
    """
    print("Used configuration:")
    print(json.dumps(OmegaConf.to_container(cfg), indent=4))