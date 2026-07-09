from sbi4atmret.config.configs import DatasetConfig
from lampe.data import H5Dataset
from pathlib import Path
from typing import Dict
import json
import logging

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, config):
        """
        Args:
            config: BaseConfig
        """
        self.config = config
        self.dataset_config = config.dataset_config
        self.training_config = config.training_config

    def _validate_metadata(self, datapath: str, dataset_name: str):
        """
        Validate that metadata.json in the dataset directory matches
        the expected simulator param names from the config.

        Raises a warning if mismatched; raises error if metadata missing
        (only warns — doesn't block loading for legacy datasets).
        """
        path = Path(datapath)
        meta_path = path / "metadata.json"

        if not meta_path.exists():
            logger.warning(
                f"No metadata.json found at {path}. "
                f"Cannot validate dataset params for '{dataset_name}'. "
                f"Re-generate with current scripts to create it."
            )
            return

        with open(meta_path) as f:
            metadata = json.load(f)

        # Check param names match config simulator names
        if hasattr(self.config, 'simulator_config') and dataset_name in self.config.simulator_config:
            expected_names = self.config.simulator_config[dataset_name].kwargs.get("names", [])
            actual_names = metadata.get("param_names", [])

            if expected_names and actual_names and expected_names != actual_names:
                logger.warning(
                    f"METADATA MISMATCH for '{dataset_name}':\n"
                    f"  Config expects: {expected_names[:5]}... ({len(expected_names)} params)\n"
                    f"  Dataset has:    {actual_names[:5]}... ({len(actual_names)} params)\n"
                    f"  This may cause silent errors during training!"
                )
            else:
                logger.info(f"  ✓ '{dataset_name}' metadata validated ({len(actual_names)} params)")

    def _build_single_loader(
        self,
        datapath: str,
        split: str,
        batch_size: int,
        shuffle: bool
    ) -> H5Dataset:
        """Build a single H5Dataset loader."""

        path = Path(datapath)
        split_path = path / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path does not exist: {split_path}")

        return H5Dataset(
            split_path,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def return_dataloaders_dict(self) -> Dict[str, Dict[str, H5Dataset]]:
        """
        Returns loaders dict and validates metadata for each dataset.

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

        # Validate metadata for each dataset
        for dataset_name, dataset_cfg in dataset_paths.items():
            self._validate_metadata(dataset_cfg.path, dataset_name)

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
        """Flatten {name: loader} → (sorted_keys, loaders_list).
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
        """Reconstruct batch_dict from keys + raw batches.
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
        return {key: batch for key, batch in zip(keys, batches)}

