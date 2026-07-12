"""
Bolometric properties: effective temperature and luminosity.

Computes T_eff and bolometric luminosity from posterior samples
by integrating simulated spectra using the Stefan-Boltzmann law.
"""

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch
from tqdm import tqdm

import astropy.units as u
import astropy.constants as c


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class BolometricResult:
    """Container for bolometric property estimates."""

    # Per-sample effective temperatures: (N,)
    teff: np.ndarray

    # Per-sample log10(L/L_sun): (N,)
    log_luminosity: np.ndarray

    # Per-sample integrated energy: (N,)
    energy: np.ndarray

    figure: Optional[Any] = None


# =========================================================
# CORE FUNCTIONS
# =========================================================

def teff_calc_and_luminosity(waves, model, dist=1.0, r_pl=1.0):
    """
    Calculate effective temperature and bolometric luminosity
    by integrating the spectrum using the Stefan-Boltzmann law.

    Args:
        waves: (D,) wavelength grid in microns
        model: (D,) flux density in W/m²/μm
        dist: distance to the object in parsecs
        r_pl: object radius in parsecs (same units as dist)

    Returns:
        teff: effective temperature in Kelvin
        log_bol_lum_solar: log10(L/L_sun)
        energy: integrated energy in Watts
    """

    def integ(waves, model):
        return np.sum(
            model[:-1] * ((dist / r_pl) ** 2)
            * (u.W / u.m**2 / u.micron)
            * np.diff(waves) * u.micron
        )

    energy = integ(waves, model)

    # Stefan-Boltzmann: T_eff = (F / σ)^(1/4)
    summed = energy / c.sigma_sb
    teff = (summed.value) ** 0.25

    # Bolometric luminosity
    surface_area = 4 * np.pi * (r_pl * 3.086e16 * u.m) ** 2
    bol_lum = surface_area * c.sigma_sb * teff ** 4

    L_sun = 3.846e26 * u.W
    bol_lum_solar = bol_lum / L_sun

    return teff, np.log10(bol_lum_solar.value), energy.value


def compute_teff_from_spectrum(
    wavelength,
    spectrum,
    R_pl,
    distance=7.34,
    scale=1e5,
):
    """
    Compute T_eff from a simulated spectrum.

    Converts the spectrum to physical flux units, selects wavelengths > 0.8 μm,
    and integrates.

    Args:
        wavelength: (D,) in microns
        spectrum: (D,) in scaled units (flux * scale)
        R_pl: planet radius in Jupiter radii
        distance: distance in parsecs (default 7.34 pc for WISE 1738)
        scale: the simulator scale factor (default 1e5)

    Returns:
        teff: effective temperature in Kelvin
    """
    from petitRADTRANS import nat_cst as nc
    import petitRADTRANS as prt

    # Select wavelengths > 0.8 μm
    mask = wavelength > 0.8
    wl = wavelength[mask]
    flux = spectrum[mask]

    # Convert from scaled flux to W/m²/μm
    # flux_physical = flux / scale * c / (λ² in cm) → convert units
    flux_physical = flux / scale * nc.c / (1e7 * 1e23 * (wl * 1e-4) ** 2)

    # Planet radius in parsecs
    r_pl_pc = 2.3168979e-9 * R_pl  # R_jup to parsec conversion

    teff, log_lum, energy = teff_calc_and_luminosity(
        wl, flux_physical / (distance / 9.9) ** 2,
        dist=distance, r_pl=r_pl_pc,
    )

    return teff, log_lum, energy


# =========================================================
# EVALUATOR
# =========================================================

class BolometricEvaluator:
    """
    Bolometric properties evaluator.

    Computes T_eff and luminosity for posterior samples by running
    them through the simulator and integrating the spectra.

    Expects shared state from BaseEvaluator via __dict__.update.
    """

    def compute_bolometric(
        self,
        posterior_samples: torch.Tensor,
        n_samples: int = 512,
        distance: float = 7.34,
    ) -> BolometricResult:
        """
        Compute T_eff and luminosity for posterior samples.

        Runs each sample through ALL simulators, concatenates the spectra
        across wavelength ranges (sorted by wavelength), then integrates
        the combined spectrum for T_eff.

        Args:
            posterior_samples: (B, D) merged posterior samples
            n_samples: how many samples to process
            distance: distance to object in parsecs

        Returns:
            BolometricResult
        """
        if posterior_samples.dim() == 1:
            posterior_samples = posterior_samples.unsqueeze(0)

        posterior_samples = posterior_samples[:n_samples]
        n_samples = posterior_samples.shape[0]  # update to actual count after truncation
        theta_dicts = self.pipe.split_theta(posterior_samples)

        # Get R_pl index from posterior names
        r_pl_idx = list(self.pipe.posterior_names).index("R_pl")

        teffs = []
        log_lums = []
        energies = []

        for i in tqdm(range(n_samples), desc="Bolometric"):
            # Run all simulators for this sample
            all_wavelengths = []
            all_spectra = []
            scale = None
            valid = True

            for sim_name, simulator in self.simulator_dict.items():
                theta_i = theta_dicts[sim_name][i].numpy()
                try:
                    output = simulator(theta_i)
                except Exception:
                    valid = False
                    break

                if output.wavelength is None or np.ndim(output.wavelength) == 0:
                    valid = False
                    break
                if output.spectrum is None or np.ndim(output.spectrum) == 0:
                    valid = False
                    break

                all_wavelengths.append(output.wavelength)
                all_spectra.append(output.spectrum)

                if scale is None:
                    scale = simulator.scale

            if not valid or len(all_wavelengths) == 0:
                teffs.append(np.nan)
                log_lums.append(np.nan)
                energies.append(np.nan)
                continue

            # Concatenate and sort by wavelength
            combined_wl = np.concatenate(all_wavelengths)
            combined_spec = np.concatenate(all_spectra)
            sort_idx = np.argsort(combined_wl)
            combined_wl = combined_wl[sort_idx]
            combined_spec = combined_spec[sort_idx]

            # Remove duplicates (overlapping wavelength regions)
            _, unique_idx = np.unique(combined_wl, return_index=True)
            combined_wl = combined_wl[unique_idx]
            combined_spec = combined_spec[unique_idx]

            R_pl = posterior_samples[i, r_pl_idx].item()

            teff, log_lum, energy = compute_teff_from_spectrum(
                combined_wl,
                combined_spec,
                R_pl=R_pl,
                distance=distance,
                scale=scale,
            )

            teffs.append(teff)
            log_lums.append(log_lum)
            energies.append(energy)

        return BolometricResult(
            teff=np.array(teffs),
            log_luminosity=np.array(log_lums),
            energy=np.array(energies),
        )
