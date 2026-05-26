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


from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

import numpy as np
from numpy import ndarray as Array

import petitRADTRANS as prt
import petitRADTRANS.retrieval.parameter as prm
from petitRADTRANS import nat_cst as nc

from joblib import Memory

from .pressure_layers import initialize_pressure_layers


MEMORY = Memory(".", mmap_mode="c", verbose=0)


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

@dataclass(frozen=True)
class SimulatorOutput:
    """
    Standardized simulator output container.
    """

    wavelength: np.ndarray

    spectrum: np.ndarray

    contribution: Optional[np.ndarray] = None

    pressures: Optional[np.ndarray] = None

    temperatures: Optional[np.ndarray] = None

    metadata: Optional[Dict[str, Any]] = None

    parameters: Optional[Dict[str, Any]] = None

# =========================================================
# SIMULATOR
# =========================================================

class Simulator:
    """
    Exoplanet emission spectrum simulator.
    """

    def __init__(
        self,
        emission_model_diseq: Callable,
        PTprofile: Callable,
        line_species: list[str]|None = None,
        cloud_species: Optional[list[str]] = None,
        rayleigh_species: list[str]|None = None,
        continuum_opacities: list[str]|None = None,
        names: list[str]|None = None,
        a: float = 4.9,
        b: float = 19,
        **kwargs,
    ):

        if names is None:
            raise ValueError("names must be provided")

        # -----------------------------------
        # model definitions
        # -----------------------------------

        self.emission_model_diseq = emission_model_diseq
        self.PTprofile = PTprofile

        # -----------------------------------
        # spectral configuration
        # -----------------------------------

        self.a = a
        self.b = b

        self.line_species = line_species 
        self.cloud_species = cloud_species or []
        self.rayleigh_species = rayleigh_species 
        self.continuum_opacities = continuum_opacities 

        self.names = names

        # -----------------------------------
        # runtime flags
        # -----------------------------------

        self.do_scat_emis = kwargs.get(
            "do_scat_emis",
            False,
        )

        # -----------------------------------
        # config
        # -----------------------------------

        default = {
            "pressure_scaling": 10,
            "pressure_simple": 100,
            "pressure_width": 3,
            "scale": 1e5,
            "N_nodes": 10,
            "N_data_sets": 9.0,
            "D_pl": 9.9 * prt.nat_cst.pc,
            "AMR": False,
            "do_scat_emis": False,
            "contribution": True,
            "PT_plot_mode": False,
            "conv": True,
        }

        self.simconfig = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        self.scale = self.simconfig["scale"]

        # -----------------------------------
        # atmosphere
        # -----------------------------------

        self.atmosphere = MEMORY.cache(
            prt.Radtrans
        )(
            line_species=self.line_species,
            cloud_species=self.cloud_species,
            rayleigh_species=self.rayleigh_species,
            continuum_opacities=self.continuum_opacities,
            wlen_bords_micron=[self.a, self.b],
            do_scat_emis=self.do_scat_emis,
        )

        levels = (
            self.simconfig["pressure_simple"]
            + len(self.atmosphere.cloud_species)
            * (
                self.simconfig["pressure_scaling"] - 1
            )
            * self.simconfig["pressure_width"]
        )#100+27

        ## sets the pressure to be an array of len(levels, here 127) between 1 to 1e9 in cgs units 
        self.atmosphere.setup_opa_structure(
            np.logspace(-6, 3, levels)
        )

        self.wavelength = (
            nc.c / self.atmosphere.freq / 1e-4
        )

    # =====================================================
    # FORWARD MODEL
    # =====================================================

    def __call__(
        self,
        theta: Array,
    ) -> SimulatorOutput:

        parameters = self._build_parameters(theta)

        # #(output P in bar-> len 1000 or 127, dict)
        pressures = initialize_pressure_layers(
            self.atmosphere.press / 1e6,
            parameters,
        )

        temperatures = self.PTprofile(
            parameters,
            pressures=pressures,
        )

        result = self._compute_emission(
            parameters,
            pressures,
            temperatures,
        )

        spectrum = self.process(
            result["spectrum"]
        )

        return SimulatorOutput(
            wavelength=result["wavelength"],
            spectrum=spectrum,
            contribution=result.get("contribution"),
            pressures=pressures,
            temperatures=temperatures,
            parameters ={
                        k: v.value
                        for k, v in parameters.items()
                    },
            metadata={
                "simulator": self.__class__.__name__,
                "cloud_species": self.cloud_species,
                "line_species": self.line_species,
            },
        )

    # =====================================================
    # INTERNALS
    # =====================================================

    def _build_parameters(
        self,
        theta: Array,
    ) -> Dict[str, prm.Parameter]:

        theta_dict = self.simconfig.copy()

        theta_dict.update(
            dict(zip(self.names, theta))
        )

        theta_dict["R_pl"] *= (
            prt.nat_cst.r_jup_mean
        )

        theta_dict["mass"] *= (
            prt.nat_cst.m_jup
        )

        return {
            k: prm.Parameter(
                name=k,
                value=v,
                is_free_parameter=False,
            )
            for k, v in theta_dict.items()
        }

    def _compute_emission(
        self,
        parameters: Dict,
        pressures: np.ndarray,
        temperatures: np.ndarray,
    ) -> Dict[str, Any]:

        if self.simconfig.get(
            "contribution",
            True,
        ):

            wv, spectrum, contribution = (
                self.emission_model_diseq(
                    self.atmosphere,
                    parameters,
                    pressures,
                    temperatures,
                )
            )

            return {
                "wavelength": wv,
                "spectrum": spectrum,
                "contribution": contribution,
            }

        wv, spectrum = self.emission_model_diseq(
            self.atmosphere,
            parameters,
            pressures,
            temperatures,
        )

        return {
            "wavelength": wv,
            "spectrum": spectrum,
        }

    # =====================================================
    # POSTPROCESSING
    # =====================================================

    def process(
        self,
        spectrum: Array,
    ) -> Array:

        if spectrum is None:
            return (
                np.ones_like(self.wavelength)
                * np.nan
            )

        if np.any(np.isnan(spectrum)):
            return (
                np.ones_like(self.wavelength)
                * np.nan
            )

        return spectrum * self.scale



    