r"""Exoplanet emission spectrum (EES) simulator.

The simulator computes an emission spectrum based on disequilibrium carbon chemistry,
equilibrium clouds and a spline temperature-pressure profile of the exoplanet atmosphere.

References:
    Retrieving scattering clouds and disequilibrium chemistry in the atmosphere of HR 8799e
    (Mollière et al., 2020)
    https://arxiv.org/abs/2006.09394

Shapes:
    theta: :math:`(16,)`
    x: :math:`(379,)`
"""

import numpy as np
import os
import pandas as pd

import petitRADTRANS as prt
import petitRADTRANS.retrieval.parameter as prm
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval.util import gaussian_prior

from joblib import Memory
from numpy import ndarray as Array
from typing import *

from scipy.interpolate import PchipInterpolator

from pathlib import Path
import os
scratch = os.environ.get('SCRATCH') 
home = os.environ.get('HOME')

import sys
sys.path.insert(0, str(Path(home) / 'WISEJ1738/sbi_ear'))
from utils.emission_model_general import emission_model_diseq, temp_model_nodes
from .pressure_layers import *

MEMORY = Memory(os.getcwd(), mmap_mode='c', verbose=0)

class Simulator(object):
    r"""Creates a EES simulator.

    Arguments:
        kwargs: Simulator settings and configuration parameters (e.g. planet distance, pressures, ...).
    """

    def __init__(self, 
                emission_model_diseq: Callable,
                PTprofile: Callable, 
                line_species: list[str] | None = None,
                cloud_species : list[str] | None = None, 
                rayleigh_species : list[str] | None = None,
                continuum_opacities : list[str] | None= None,
                names : list[str] | None = None,
                a:float = 4.9, 
                b:float = 19, 
                **kwargs):        
                 
        super().__init__()

        # Scalars
        self.a = a
        self.b = b

        # Lists
        self.line_species = line_species
        self.rayleigh_species = rayleigh_species 
        self.continuum_opacities = continuum_opacities 
        self.names = names
        if self.names is None:
            raise ValueError("names must be provided")

        #boolean
        self.do_scat_emis = kwargs.get("do_scat_emis", False)
        
        # Cloud species handling
        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        
        self.emission_model_diseq = emission_model_diseq
        self.PTprofile = PTprofile

        # Config
        default = {
                    'pressure_scaling': 10,
                    'pressure_simple': 100,
                    'pressure_width': 3,
                    'scale': 1e5, #1e16,
                    'N_nodes' : 10,
                    'N_data_sets': 9.000,
                    'D_pl' : 9.9 * prt.nat_cst.pc, 
                    'AMR' : False,
                    'do_scat_emis' : False,
                    'contribution' : True,
                    'PT_plot_mode' : False,
                    'conv' : True
                }
        

        self.config = {
            k: kwargs.get(k, v) for k, v in default.items()
        }
        self.scale = self.config.pop('scale')
        

        self.atmosphere = MEMORY.cache(prt.Radtrans)(
            line_species= self.line_species,
            cloud_species= self.cloud_species, 
            rayleigh_species= self.rayleigh_species,
            continuum_opacities= self.continuum_opacities,
            wlen_bords_micron=[self.a, self.b], 
            do_scat_emis= self.do_scat_emis,
        )

        levels = (
            self.config['pressure_simple'] + len(self.atmosphere.cloud_species) *
            (self.config['pressure_scaling'] - 1) * self.config['pressure_width']
        ) #100+27

        ## sets the pressure to be an array of len(levels, here 127) between 1 to 1e9 in cgs units 
        self.atmosphere.setup_opa_structure(np.logspace(-6, 3, levels))  
        self.wavelength = nc.c/self.atmosphere.freq/1e-4


    def __call__(self, theta: Array) -> Array:

        theta_dict = self.config.copy()
        theta_dict.update(dict(zip(self.names, theta)))
        theta_dict['R_pl'] = theta_dict['R_pl'] * prt.nat_cst.r_jup_mean
        theta_dict['mass'] = theta_dict['mass'] * prt.nat_cst.m_jup

        self.parameters = {
            k: prm.Parameter(name=k, value=v, is_free_parameter=False)
            for k, v in theta_dict.items()
        }

        if self.config.get("contribution", True):
            wv, x, contr_em = emission_spectrum(self.atmosphere, self.parameters, self.emission_model_diseq, self.PTprofile)
            x = self.process(x)
            return wv, x, contr_em 
        else:
            wv, x = emission_spectrum(self.atmosphere, self.parameters, self.emission_model_diseq, self.PTprofile)
            x = self.process(x)
            return wv, x

    def process(self, x: Array) -> Array:
        r"""Processes spectra into network-friendly inputs."""
        if np.any(x) == None: 
            return np.ones_like(self.wavelength) * np.nan
        else:
            return x * self.scale

def emission_spectrum(
    atmosphere: prt.Radtrans,
    parameters: Dict,
    emission_model_diseq : Callable, 
    PTprofile: Callable, 
) -> Array:
    r"""Simulates the emission spectrum of an exoplanet."""

    pressures = initialize_pressure_layers(atmosphere.press/1e6,parameters) #(output P in bar-> len 1000 or 127, dict)
    temperatures = PTprofile(parameters, pressures = pressures)
    
    if parameters['contribution']:
        wv, spectrum, contr_em = emission_model_diseq(atmosphere, parameters, pressures, temperatures)
        return wv, spectrum, contr_em
    
    else:
        wv, spectrum = emission_model_diseq(atmosphere, parameters, pressures, temperatures)
        return wv, spectrum



