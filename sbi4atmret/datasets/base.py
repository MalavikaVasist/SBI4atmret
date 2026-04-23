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
            {
                "train": {
                    "miri": loader,
                    "gemini": loader,
                    "hst": loader
                },
                "valid": {...},
                "test": {...}
            }
        """

        dataset_paths = self.dataset_config.dataset_path
        batch_size = self.training_config.batch_size
        shuffle = self.dataset_config.shuffle

        dataloaders = {
            "train": {},
            "valid": {},
            "test": {}
        }

        # loop over condition (e.g. cloudfree)
        for condition, instruments in dataset_paths.items():

            for instrument_name, inst_cfg in instruments.items():
                datapath = inst_cfg.path

                for split in ["train", "valid", "test"]:
                    loader = self._build_single_loader(
                        datapath=datapath,
                        split=split,
                        batch_size=batch_size if split != "test" else 16,
                        shuffle=shuffle if split != "test" else False
                    )

                    dataloaders[split][instrument_name] = loader

        return dataloaders