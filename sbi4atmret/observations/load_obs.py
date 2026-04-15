import numpy as np
import pandas as pd
from pathlib import Path

from ..utils.config import Config

OBS_BASE_DIR = Path.home() / "SBI4exoplanets" / "observations"


def _normalize_tag(tag):
    return None if tag in (None, "None") else tag


def _read_spectrum(file_path: Path):
    '''
    Read CSV file as pandas DataFrame.
    '''
    frame = pd.read_csv(file_path, header=0, sep=",")
    return (
        np.asarray(frame.iloc[:, 0]),
        np.asarray(frame.iloc[:, 1]),
        np.asarray(frame.iloc[:, 2]),
    )


def _scale_flux_and_sigma(flux, sigma, distance, distance_sim, scale_factor):
    '''
    Scale flux and sigma based on distance.
    scaling the observations as if they were seen from D_sim pc to use simulations that were simulated for D_sim pc 
    '''
    factor = (distance / distance_sim) ** 2
    return flux * scale_factor * factor, sigma * factor


class ObservationLoader:
    '''
    ['WISE0855', 'WISE0458', 'ROSS458C', 'WISEJ1828', 'WISEJ1738']
    [2.28, 14, 11.5, 9.9, 7.34]

    #apply whatever source, instruments to return as a dictionary as 
    
    when source "WISEJ1738" and instrument is ["miri", "hst"] and tag is [None] 
    observation= {
                    "WISEJ1738": {
                                "miri" : {
                                        "wlen" : obs_wlen_miri,
                                        "flux" : obs_miri,
                                        "err" : sigmaM,
                                        },

                                "hst" : {
                                        "wlen" : obs_wlen_hst,
                                        "flux" : obs_hst,
                                        "err" : sigmaH,
                                        },
                                }
            }
    OR 
    when source "WISEJ1738" and instrument is ["miri"] and tag is ["Helena_rebinning_all_ch_together"]
    observation= {
                    "WISEJ1738": {
                                "rebinning_all_ch_together" : {
                                                                    "miri" : {
                                                                            "wlen" : obs_wlen_miri,
                                                                            "flux" : obs_miri,
                                                                            "err" : sigmaM,
                                                                    },
                                                                    "hst" : {
                                                                            "wlen" : obs_wlen_hst,
                                                                            "flux" : obs_hst,
                                                                            "err" : sigmaH,
                                                                            },
                                                                    },
                                                            }
                                    }
                                }
    '''
    def __init__(self, config: Config):
        self.source = config["source"]
        self.instruments = config["instruments"]
        self.tag = _normalize_tag(config.get("tag"))
        self.distance = config["D"]
        self.sim_distance = config["simulator"]["D_pl"]
        self.scale_factor = config["simulator"]["scale"]
        self.base_path = OBS_BASE_DIR / self.source
        self.spectrum_filename = config.get("spectrum_filename", "spectrum.csv")

        self.observation = {self.source: {}}
        self.target = self.observation[self.source]

    def load(self):
        if self.tag and "simulation" in self.tag:
            return self._load_simulation()
        return self._load_real_observations()

    def _load_simulation(self):
        '''
        [mostprobCF_simulation0_noisefree, "mostprobCF_simulation1", "mostprobCF_simulation2", "mostprobCF_simulation3",
        mostprobCLavg_simulation1, mostprobCLavg_simulation2, mostprobCLavg_simulation3, mostprobCLavg_simulation0_noisefree]

        file_map = {
#             'simulation0': 'mostprobCLavg_simulation0_noisefree.csv',
#             'simulation1': 'mostprobCLavg_simulation1.csv',
#             'simulation2': 'mostprobCLavg_simulation2.csv',
#             'simulation3': 'mostprobCLavg_simulation3.csv',
#             'simulation0_cf': 'mostprobCF_simulation0_noisefree.csv',
#             'simulation1_cf': 'mostprobCF_simulation1.csv',
#             'simulation2_cf': 'mostprobCF_simulation2.csv',
#             'simulation3_cf': 'mostprobCF_simulation3.csv'
#         }
        '''
        sim_file = self.base_path / "simulations" / f"{self.tag}.csv"
        if not sim_file.exists():
            raise FileNotFoundError(f"Simulation file not found: {sim_file}")

        x_star = pd.read_csv(sim_file).iloc[0]
        full_flux = np.asarray(x_star)

        # Load wlen and err for all instruments
        wlens = {}
        errs = {}
        for i, instrument in enumerate(self.instruments):
            wlen, _, err = _read_spectrum(self.base_path / instrument / self.spectrum_filename[i])
            wlens[instrument] = wlen
            errs[instrument] = err

        # Concatenate all wavelengths in instrument order
        all_wlen = np.concatenate([wlens[inst] for inst in self.instruments])
        
        # Get the sort indices for the concatenated wavelengths
        sort_idx = np.argsort(all_wlen)
        
        # The full_flux is assumed to be in the order of sorted wavelengths
        # To get back to the concatenated order (instrument order), use inverse sort
        inverse_sort_idx = np.argsort(sort_idx)
        flux_concat = full_flux[inverse_sort_idx]
        
        # Now slice the flux for each instrument
        cum_len = 0
        self.target[self.tag] = {}
        for inst in self.instruments:
            len_inst = len(wlens[inst])
            flux_inst = flux_concat[cum_len : cum_len + len_inst]
            # Scale the error
            err_scaled = errs[inst] * (self.distance / self.sim_distance) ** 2
            self.target[self.tag][inst] = {
                                            "wlen": wlens[inst],
                                            "flux": flux_inst,
                                            "err": err_scaled
            }
            cum_len += len_inst

        return self.target

    def _load_real_observations(self):
        
        if self.tag:
            target = self.target[self.tag]
        else:
            target = self.target

        for i, instrument in enumerate(self.instruments):
            file_path = self.base_path / instrument
            if self.tag:
                file_path = file_path / self.tag
            file_path = file_path / self.spectrum_filename[i]

            if not file_path.exists():
                raise FileNotFoundError(f"Observation file not found: {file_path}")

            wlen, flux, sigma = _read_spectrum(file_path)
            flux, sigma = _scale_flux_and_sigma(flux, sigma, self.distance, self.sim_distance, self.scale_factor)
            target[instrument] = {"wlen": wlen, "flux": flux, "err": sigma}

        return target


def load_observations(config: Config):
    return ObservationLoader(config).load()
    
