from config import DatasetConfig
from lampe.data import H5Dataset
from pathlib import Path

class Dataset:
    def __init__(self, config: DatasetConfig):
        self.dataset_config = config

    def build_dataloader_batchwise(
        self,
        datapath: str,
        split: str, 
        batch_size : int, 
        shuffle: str) -> H5Dataset:

        if split == 'test':
            return H5Dataset(Path(datapath) / split/ '.h5', 
                               batch_size = 16, 
                               shuffle = shuffle)
        else:
            return H5Dataset(Path(datapath) / split/ '.h5', 
                               batch_size = batch_size, 
                               shuffle = shuffle)


    def return_dataloaders(
        self,
    ) -> List[H5Dataset]: ## [train, valid, test]

    dataloader_list = []
    data_config = self.config.dataset_config

    for atm in data_config.dataset_path.keys(): 
        for inst in data_config.dataset_path[atm].keys():
            for split in ['train', 'valid', 'test']:
                dataloader_list.append(build_dataloader_batchwise(
                                    datapath = data_config.dataset_path[atm][inst].path, 
                                    split = split, 
                                    batch_size = self.config.training.batch_size, 
                                    shuffle= data_config.shuffle
            ))
                
    return dataloader_list

    