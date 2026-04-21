class Dataset:
    def __init__(self, config: BaseConfig):
        self.config = config

    def load_dataset_batchwise(
        self,
        config: BaseConfig, 
        dataset_name: str,
        split: str = 'train'
    ) -> H5Dataset:
        """
        Load dataset in batches using H5Dataset.
        
        Args:
            config: BaseConfig instance with dataset configuration
            dataset_name: Name of the dataset to load
            split: Dataset split ('train', 'valid', or 'test')
            
        Returns:
            H5Dataset instance
            
        Raises:
            ValueError: If config is invalid or dataset path doesn't exist
        """
        if not isinstance(config, BaseConfig):
            raise TypeError(f"Expected BaseConfig, got {type(config)}")
        
        if config.observation is None:
            raise ValueError("Observation configuration not found in config")
        
        # Construct dataset path
        dataset_config = config.observation
        if not hasattr(dataset_config, 'dataset_path') or dataset_config.dataset_path is None:
            raise ValueError("Dataset path not configured in observation config")
        
        # Get batch size from training config
        batch_size = 64  # Default batch size
        if config.training and hasattr(config.training, 'batch_size'):
            batch_size = config.training.batch_size
            if isinstance(batch_size, list):
                batch_size = batch_size[0]
        
        # Construct full path
        dataset_path_dict = dataset_config.dataset_path
        if isinstance(dataset_path_dict, dict):
            # Try to get path for the dataset
            path = dataset_path_dict.get(split, {}).get(dataset_name, None)
            if path is None:
                # Try alternate structure
                for key, val in dataset_path_dict.items():
                    if isinstance(val, dict) and dataset_name in val:
                        path = val[dataset_name]
                        break
        else:
            path = str(dataset_path_dict)
        
        if path is None:
            raise ValueError(f"Dataset path for '{dataset_name}' not found in config")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        
        logger.info(f"Loading dataset from {path} with batch_size={batch_size}...")
        
        try:
            dataset = H5Dataset(path, batch_size=batch_size)
            logger.info(f"Dataset loaded successfully: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

