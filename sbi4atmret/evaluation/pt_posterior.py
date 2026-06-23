"""
PT profile posterior evaluation with emission contribution function overlay.

Produces a plot of the posterior PT profile with:
- Credible-region bands on the T-P plane
- Spectrally-weighted emission contribution function
- Condensation curves
- Optional: overlay from multiple simulators/posteriors
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

from sbi4atmret.utils.general import instrument_from_simname, find_map_sample
from sbi4atmret.utils.plotting_utils import legends


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class PTResult:
    """
    Container for PT profile evaluation outputs.

    Holds per-instrument PT data keyed by sim_name,
    plus the combined figure.
    """

    # {sim_name: {"temperatures", "pressures", "contribution",
    #             "spectrum", "wavelength", "contr_em_weighted",
    #             "spectral_weights", "map_index", "map_sample"}}
    pt_data: dict

    # list of simulator names in plot order
    sim_names: List[str]

    # the combined figure
    figure: Optional[Any] = None


# =========================================================
# EVALUATOR
# =========================================================

class PTEvaluator:
    """
    PT profile posterior evaluator.

    Expects to be initialized with a shared state dict
    from BaseEvaluator (via __dict__.update).
    """

    # =====================================================
    # COMPUTE PT PROFILES
    # =====================================================

    def compute_pt_profiles(
        self,
        posterior_samples: torch.Tensor,
        sim_name: str,
        pt_dict_path: Path,
        n_samples: int = 256,
    ) -> dict:
        """
        Get PT profiles for a simulator.

        Checks if pt_dict_path already has cached temperatures/pressures
        from ConsistencyEvaluator. Contribution is always computed fresh
        from the MAP sample.

        Args:
            posterior_samples: (B, D) merged posterior samples
            sim_name: which simulator to use
            pt_dict_path: path to the cached pt_dict.pt file
            n_samples: how many samples for PT bands

        Returns:
            dict with keys: temperatures, pressures, contribution,
            spectrum, wavelength, map_index, map_sample
        """

        # Find MAP sample
        map_index, map_sample = find_map_sample(
            self.net, posterior_samples, self.x_obs, device=self.device
        )

        # Run MAP sample through simulator for contribution
        map_theta_dict = self.pipe.split_theta(map_sample.unsqueeze(0))
        simulator = self.simulator_dict[sim_name]
        map_theta_i = map_theta_dict[sim_name][0].numpy()
        map_output = simulator(map_theta_i)

        # Check if temperatures were already cached by consistency
        if pt_dict_path.exists():
            all_pt = torch.load(pt_dict_path, map_location="cpu")
            if sim_name in all_pt:
                pt_data = all_pt[sim_name]
                pt_data["contribution"] = map_output.contribution
                pt_data["spectrum"] = map_output.spectrum
                pt_data["wavelength"] = map_output.wavelength
                pt_data["map_index"] = map_index
                pt_data["map_sample"] = map_sample
                return pt_data

        # Not cached — run simulator from scratch for PT bands
        if posterior_samples.dim() == 1:
            posterior_samples = posterior_samples.unsqueeze(0)

        theta_dict = self.pipe.split_theta(posterior_samples[:n_samples])

        temperatures_list = []

        for i in tqdm(range(theta_dict[sim_name].shape[0]), desc=f"PT {sim_name}"):
            theta_i = theta_dict[sim_name][i].numpy()
            output = simulator(theta_i)

            if output.temperatures is not None:
                temperatures_list.append(output.temperatures)

        temperatures = np.stack(temperatures_list)

        return {
            "temperatures": temperatures,
            "pressures": map_output.pressures,
            "contribution": map_output.contribution,
            "spectrum": map_output.spectrum,
            "wavelength": map_output.wavelength,
            "map_index": map_index,
            "map_sample": map_sample,
        }

    # =====================================================
    # SPECTRAL WEIGHTS & CONTRIBUTION
    # =====================================================

    def compute_spectral_weights(self, wavelength, spectrum):
        """
        Compute spectral weights from frequency spacing.

        Args:
            wavelength: (D,) in microns
            spectrum: (D,) flux

        Returns:
            spectral_weights: (D,)
        """
        from petitRADTRANS import nat_cst as nc

        nu = nc.c / (wavelength * 1e-4)
        mean_diff_nu = -np.diff(nu)
        diff_nu = np.zeros_like(nu)
        diff_nu[:-1] = mean_diff_nu
        diff_nu[-1] = diff_nu[-2]

        spectral_weights = spectrum * diff_nu / np.sum(spectrum * diff_nu)

        return spectral_weights

    def compute_contribution_weights(self, pressures, contribution, spectral_weights):
        """
        Compute pressure-normalized, spectrally-weighted contribution.

        Args:
            pressures: (n_pressures,) in bar
            contribution: (n_pressures, D)
            spectral_weights: (D,)

        Returns:
            contr_em_weighted: (n_pressures,) normalized 0-1
        """
        pressure_weights = np.diff(np.log10(pressures))
        weights = np.ones_like(pressures)
        weights[:-1] = pressure_weights
        weights[-1] = weights[-2]
        weights = weights / np.sum(weights)
        weights = weights.reshape(-1, 1)

        contr_em0 = contribution / weights

        contr_em = np.sum(contr_em0 * spectral_weights, axis=1) / np.sum(contr_em0)
        contr_em_weighted = contr_em / np.max(contr_em)

        return contr_em_weighted

    # =====================================================
    # CONDENSATION CURVES
    # =====================================================

    @staticmethod
    def condensation_curves(pressures, feh=0.0):
        """
        Compute condensation T-P curves for common species.

        Args:
            pressures: (n_pressures,) in bar
            feh: metallicity [Fe/H]

        Returns:
            dict {species_name: {"T": array, "color": str}}
        """
        log_p = pressures

        curves = {
            r"H$_2$O": {
                "T": 1e4 / (38.84 - 3.93 * feh - 3.83 * np.log10(log_p)
                            - 0.2 * feh * np.log10(log_p)),
                "color": "blue",
            },
            r"NH$_3$": {
                "T": 1e4 / (68.02 - 6.19 * feh - 6.31 * np.log10(log_p)),
                "color": "purple",
            },
            r"Na$_2$S": {
                "T": 1e4 / (10.045 - 0.72 * np.log10(log_p) - 1.08 * feh),
                "color": "green",
            },
            r"KCl": {
                "T": 1e4 / (12.479 - 0.879 * np.log10(log_p) - 0.879 * feh),
                "color": "red",
            },
        }

        return curves

    # =====================================================
    # PLOTTING — SINGLE POSTERIOR
    # =====================================================

    def plot_pt(
        self,
        temperatures,
        pressures,
        contr_em_weighted=None,
        color="steelblue",
        creds=(0.997, 0.955, 0.683),
        alpha=(0.0, 0.9),
        figsize=(6, 7),
        xlim=(0, 4000),
        feh=0.0,
        frac=0.6,
        show_condensation=True,
        show_contribution=True,
        show_legend=True,
        ax=None,
        fig=None,
    ):
        """
        Plot a single posterior PT profile with contribution and condensation.

        Args:
            temperatures: (B, n_pressures)
            pressures: (n_pressures,) in bar
            contr_em_weighted: (n_pressures,) normalized 0-1, or None
            color, creds, alpha: styling
            figsize, xlim: layout
            feh: metallicity for condensation curves
            frac: max whitening for contribution overlay
            show_condensation, show_contribution, show_legend: toggles
            ax, fig: existing axes/figure (creates new if None)
        """
        from lampe.plots import LinearAlphaColormap

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Colormap setup
        creds_arr = np.sort(np.asarray(creds))[::-1]
        creds_arr = np.append(creds_arr, 0)
        levels = (creds_arr - creds_arr.min()) / (creds_arr.max() - creds_arr.min())
        levels = (levels[:-1] + levels[1:]) / 2
        cmap = LinearAlphaColormap(color, levels=creds_arr, alpha=alpha)

        # PT credible regions
        for q, l in zip(creds_arr[:-1], levels):
            left, right = np.quantile(
                temperatures, [0.5 - q / 2, 0.5 + q / 2], axis=0
            )
            ax.fill_betweenx(
                pressures, left, right,
                color=cmap(l), linewidth=0,
            )

        # Contribution function overlay (white fading)
        if show_contribution and contr_em_weighted is not None:
            tlims = (np.min(temperatures) * 0.97, np.max(temperatures) * 1.03)
            contr_interp = interp1d(pressures, contr_em_weighted)

            for i_p in range(len(pressures) - 1):
                mean_press = (pressures[i_p + 1] + pressures[i_p]) / 2.0
                ax.fill_between(
                    tlims,
                    pressures[i_p + 1],
                    pressures[i_p],
                    color="white",
                    alpha=min(1.0 - contr_interp(mean_press), frac),
                    linewidth=0,
                    rasterized=True,
                    zorder=4,
                )

            # Plot contribution curve
            ax.plot(
                contr_em_weighted * (tlims[1] - tlims[0]) + tlims[0],
                pressures,
                "--",
                color=color,
                linewidth=1,
                zorder=5,
            )

        # Condensation curves
        if show_condensation:
            curves = self.condensation_curves(pressures, feh=feh)
            for name, info in curves.items():
                ax.plot(
                    info["T"], pressures,
                    lw=2, color=info["color"],
                    label=f"{name} condensation",
                )

        # Axis formatting
        ax.set_yscale("log")
        ax.set_ylim([pressures[-1] * 1.03, pressures[0] / 1.03])
        ax.set_xlim(xlim)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Pressure [bar]")

        if show_legend:
            ax.legend(loc="upper right", fontsize=10)

        return fig

    # =====================================================
    # PLOTTING — MULTIPLE POSTERIORS OVERLAID
    # =====================================================

    def plot_pt_multi(
        self,
        pt_data_list: List[dict],
        colors: List[str] = None,
        creds_list: List[tuple] = None,
        alpha_list: List[tuple] = None,
        figsize=(5, 5),
        xlim=(0, 4000),
        feh=0.0,
        frac=0.6,
        show_condensation=True,
        show_contribution=True,
        show_legend_list: List[bool] = None,
        rcparams=None,
    ):
        """
        Overlay multiple PT posteriors on a single plot.

        Generalized version of pt_plotting_wcondcurv from f.py.

        Args:
            pt_data_list: list of dicts, each with keys:
                - temperatures: (B, n_pressures)
                - pressures: (n_pressures,)
                - contr_em_weighted: (n_pressures,) or None
            colors: list of color strings per posterior
            creds_list: list of cred tuples per posterior
            alpha_list: list of alpha tuples per posterior
            figsize: figure size
            xlim: (Tmin, Tmax)
            feh: metallicity for condensation curves
            frac: max whitening for contribution overlay
            show_condensation: plot condensation curves (once)
            show_contribution: overlay contribution per posterior
            show_legend_list: list of bools, show legend per posterior
            rcparams: matplotlib rcParams overrides

        Returns:
            (fig, ax)

        Example:
            fig, ax = pt_eval.plot_pt_multi(
                pt_data_list=[
                    {"temperatures": T1, "pressures": P, "contr_em_weighted": C1},
                    {"temperatures": T2, "pressures": P, "contr_em_weighted": C2},
                ],
                colors=["steelblue", "orange"],
            )
        """

        n = len(pt_data_list)

        if colors is None:
            colors = ["steelblue"] * n
        if creds_list is None:
            creds_list = [(0.997, 0.955, 0.683)] * n
        if alpha_list is None:
            alpha_list = [(0.0, 0.9)] * n
        if show_legend_list is None:
            show_legend_list = [True] + [False] * (n - 1)

        params = rcparams or {"axes.labelsize": 14}
        plt.rcParams.update(params)

        fig, ax = plt.subplots(figsize=figsize)

        for i, pt_data in enumerate(pt_data_list):

            temperatures = pt_data["temperatures"]
            pressures = pt_data["pressures"]
            contr_em_weighted = pt_data.get("contr_em_weighted", None)

            # Remove inf samples
            if isinstance(temperatures, torch.Tensor):
                mask = ~torch.any(torch.isinf(torch.from_numpy(temperatures)), dim=-1)
                temperatures = temperatures[mask.numpy()]

            self.plot_pt(
                temperatures,
                pressures,
                contr_em_weighted=contr_em_weighted,
                color=colors[i],
                creds=creds_list[i],
                alpha=alpha_list[i],
                xlim=xlim,
                feh=feh,
                frac=frac,
                show_condensation=(show_condensation and i == 0),
                show_contribution=show_contribution,
                show_legend=show_legend_list[i],
                ax=ax,
                fig=fig,
            )

        return fig, ax
