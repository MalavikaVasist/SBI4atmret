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
        self.pipe = self.config.build_pipe()
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


    def return_dataloaders_dict(self) -> Dict[str, Dict[str, H5Dataset]]:
        """
        Returns

        {
            "train": {
                "cloudfree_miri": loader,
                "cloudfree_gemini": loader,
                "cloudfree_hst": loader,
            },
            "valid": {...},
            "test": {...}
        }
        """

        dataset_paths = self.dataset_config.dataset_path
        batch_size = self.training_config.batch_size
        shuffle = self.dataset_config.shuffle

        dataloaders = {}

        for split in ["train", "valid", "test"]:
            dataloaders[split] = {}

            for dataset_name, dataset_cfg in dataset_paths.items():
                loader = self._build_single_loader(
                    datapath=dataset_cfg.path,
                    split=split,
                    batch_size=batch_size if split != "test" else 16,
                    shuffle=shuffle if split != "test" else False,
                )

                dataloaders[split][dataset_name] = loader

        return dataloaders


    @staticmethod
    def flatten_loaders(loaders_dict):
        """
        Input:
            {
                "cloudfree_miri": loader,
                "cloudfree_gemini": loader,
                "cloudfree_hst": loader
            }

        Returns:
            keys    = ["cloudfree_gemini", "cloudfree_hst", "cloudfree_miri"]
            loaders = [loader, loader, loader]
        """

        keys = sorted(loaders_dict.keys())
        loaders = [loaders_dict[k] for k in keys]

        return keys, loaders

    @staticmethod
    def reconstruct_batch(keys, batches):
        """
        Args:
            keys:
                ["cloudfree_gemini",
                "cloudfree_hst",
                "cloudfree_miri"]

            batches:
                [batch1, batch2, batch3]

        Returns:
            {
                "cloudfree_gemini": batch1,
                "cloudfree_hst": batch2,
                "cloudfree_miri": batch3
            }
        """

        return {
            key: batch
            for key, batch in zip(keys, batches)
        }

