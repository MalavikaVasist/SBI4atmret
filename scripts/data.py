from pathlib import Path
from lampe.data import H5Dataset
from ..sbi4exoplanets.utils.config import Config
from ..sbi4exoplanets.observations.load_obs import load_observations


def load_observations_data(config: Config):
    observation = load_observations(config)
    return observation


def load_datasets(config: Config, scratch: str):
    '''
    Load datasets for all atmospheric types and instruments as specified in the config.
    Returns: 
    {
        'cloudfree': {
            'hst': {'train': H5Dataset, 'valid': H5Dataset, 'test': H5Dataset},
            'gemini': {'train': H5Dataset, 'valid': H5Dataset, 'test': H5Dataset},
            'miri': {'train': H5Dataset, 'valid': H5Dataset, 'test': H5Dataset},
            ...
        },
        'cloudy': {
            ...
        },
        ...
    }
    '''
    datasets = {}
    batch_size = config["ML_model_configs"]["batch_size"]
    for atm_type in config['simulator']["type"]:
        datasets[atm_type] = {}
        for instrument in config['instruments']:
            try:
                path = config["sim_paths"][atm_type][instrument.lower()]
            except KeyError:
                raise ValueError(f"Missing path for {instrument} and type {atm_type}")

            datasets[atm_type][instrument] = {
                'train': H5Dataset(Path(scratch) / path / 'train.h5', batch_size=batch_size),
                'valid': H5Dataset(Path(scratch) / path / 'valid.h5', batch_size=batch_size),
                'test': H5Dataset(Path(scratch) / path / 'test.h5', batch_size=16),
            }
    return datasets