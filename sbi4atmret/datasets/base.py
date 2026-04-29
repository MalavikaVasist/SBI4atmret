from config import DatasetConfig
from lampe.data import H5Dataset
from pathlib import Path
from typing import Dict


class Dataset:
    def __init__(self, config):
        """
        Args:
            config: BaseConfig
        """
        self.config = config
        self.dataset_config = config.dataset_config
        self.training_config = config.training_config

    def _build_single_loader(
        self,
        datapath: str,
        split: str,
        batch_size: int,
        shuffle: bool
    ) -> H5Dataset:
        """
        Build a single H5Dataset loader.
        """

        path = Path(datapath)

        # Assume structure: datapath/train/*.h5 etc.
        split_path = path / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path does not exist: {split_path}")

        # If H5Dataset expects a file:
        # adjust this if needed (e.g. "data.h5")
        return H5Dataset(
            split_path,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def return_dataloaders(self) -> Dict[str, Dict[str, H5Dataset]]:

        """
        Returns:
            {train:{
                cloudfree: {
                    "miri": loader,
                    "gemini": loader,
                    "hst": loader
                },
                cloudy :{
                    "miri": loader,
                    "gemini": loader,
                    "hst": loader
                },
            
            "valid": {...},
            "test": {...}
            }}

        """

        dataset_paths = self.dataset_config.dataset_path
        batch_size = self.training_config.batch_size
        shuffle = self.dataset_config.shuffle

        dataloaders = {}

        for split in ["train", "valid", "test"]:
            dataloaders[split] = {}

            for condition, instruments in dataset_paths.items():

                # initialize condition inside split
                if condition not in dataloaders[split]:
                    dataloaders[split][condition] = {}

                for instrument_name, inst_cfg in instruments.items():
                    datapath = inst_cfg.path

                    loader = self._build_single_loader(
                        datapath=datapath,
                        split=split,
                        batch_size=batch_size if split != "test" else 16,
                        shuffle=shuffle if split != "test" else False
                    )

                    dataloaders[split][condition][instrument_name] = loader

        return dataloaders

    def flatten_loaders(loaders_dict):
        """
        train/test/valid loaders 
        loaders_dict= 
        "cloudfree": {
                "miri": "loader".
                "gemini": "loader",
                "hst": "loader"
            }
        "cloudy": {
                "miri": "loader",
                "gemini": "loader",
                "hst": "loader"
            }
        
        """
        keys = []
        loaders = []

        for outer_k, inner_dict in loaders_dict.items():
            for inner_k, loader in inner_dict.items():
                keys.append((outer_k, inner_k))
                loaders.append(loader)

        return keys, loaders

    def reconstruct_batch(keys, batches):
        """
        args: 
        keys : (cloudfree, miri), (cloudfree, hst)...
        batches : (theta, x)

        returns: one batch from each loader dict
        batch_dict = {
                    "cloudfree": {
                            "miri": (theta,x), .
                            "gemini":  (theta,x),
                            "hst":  (theta,x)
                        }
                    "cloudy": {
                            "miri":  (theta,x),
                            "gemini":  (theta,x),
                            "hst":  (theta,x)
                        
                        }
        """
        batch_dict = {}

        for (outer_k, inner_k), batch in zip(keys, batches):
            batch_dict.setdefault(outer_k, {})[inner_k] = batch

        return batch_dict



