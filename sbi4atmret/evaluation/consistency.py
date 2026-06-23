from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sbi4atmret.utils.general import instrument_from_simname
from sbi4atmret.utils.plotting_utils import legends


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class ConsistencyResult:

    posterior_samples: Optional[torch.Tensor]

    predictive_dict: dict

    residuals_dict: dict

    merged_prediction: np.ndarray

    merged_wavelength: np.ndarray

    merged_residuals: np.ndarray

    figure: Optional[Any] = None


# =========================================================
# EVALUATOR
# =========================================================

class ConsistencyEvaluator:

    """
    Posterior predictive consistency checks.

    Runs posterior samples through all simulators,
    merges predictions into observation space,
    and compares against observed spectra.

    Expects to be initialized with a shared state dict
    from BaseEvaluator (via __dict__.update).
    """

    # =====================================================
    # POSTERIOR SAMPLING
    # =====================================================

    def consistency_samples(
        self,
        n_samples: int = 512,
    ) -> torch.Tensor:

        idx = torch.randperm(self.theta.shape[0])[:n_samples]
        theta = self.theta[idx].cpu()

        return theta

    # =====================================================
    # PREDICTIVE SIMULATION
    # =====================================================

    def compute_posterior_predictive(
        self,
        posterior_samples: torch.Tensor,
    ) -> dict:

        """
        Simulate spectra from all simulators.

        Returns a dict matching the pipe's batch_dict format:
            {sim_name: (theta, x)}
        where theta is (B, D_inst) and x is (B, D_spec).

        The returned dict has pipe spectral transforms and noise applied.

        Also collects PT profiles (temperatures, pressures) per simulator
        for reuse by PTEvaluator. Contribution function is computed
        separately by PTEvaluator using the MAP sample.
        """

        # split_theta expects (B, D) — add batch dim if needed
        if posterior_samples.dim() == 1:
            posterior_samples = posterior_samples.unsqueeze(0)

        theta_dict = self.pipe.split_theta(posterior_samples)

        predictive_dict = {}
        pt_dict = {}  # {sim_name: {temperatures, pressures}}

        for sim_name, simulator in self.simulator_dict.items():

            spectra = []
            temperatures_list = []
            pressures = None

            for i in tqdm(
                range(theta_dict[sim_name].shape[0]),
                desc=f"{sim_name}",
            ):

                theta_i = theta_dict[sim_name][i].numpy()
                output = simulator(theta_i)

                spectra.append(
                    torch.from_numpy(output.spectrum).float()
                )

                # Collect PT data
                if output.temperatures is not None:
                    temperatures_list.append(output.temperatures)
                    if pressures is None:
                        pressures = output.pressures

            x = torch.stack(spectra)  # (B, D_spec)

            predictive_dict[sim_name] = (
                theta_dict[sim_name],  # (B, D_inst)
                x,                     # (B, D_spec)
            )

            # Store PT profiles for this simulator
            if temperatures_list:
                pt_dict[sim_name] = {
                    "temperatures": np.stack(temperatures_list),
                    "pressures": pressures,
                }

        # Save PT dict for reuse by PTEvaluator
        self._pt_dict = pt_dict

        # Apply spectral transforms (no modify_theta) and noise
        predictive_dict = self.batch_processor.process(
            predictive_dict, mode="eval", add_noise=True
        )

        return predictive_dict

    # =====================================================
    # RESIDUALS
    # =====================================================

    def compute_residuals(self, predictive_dict: dict) -> dict:
        """
        Compute per-instrument residuals: (x_pred - x_obs) / (sigma * scale).
        """

        residuals = {}

        for sim_name in self.simulator_dict.keys():

            theta, x_pred = predictive_dict[sim_name]

            instrument_name = instrument_from_simname(sim_name)
            x_obs = self.observation.observation_dict[instrument_name]

            sigma = self.noise.compute_sigma(theta, sim_name)
            res = (x_pred - x_obs) / (sigma * self.domain.scale)

            residuals[sim_name] = (theta, res)

        return residuals

    # =====================================================
    # MERGING
    # =====================================================

    def combine_predictives(self, predictive_dict: dict):
        """
        Merge processed per-instrument predictions into observation space.
        """
        _, merged_x = self.batch_processor.merge(
            predictive_dict, mode="eval"
        )

        merged_wavelength = self.observation.full_wavelength

        return merged_wavelength, merged_x.cpu().numpy()

    def combine_residuals(self, residuals_dict: dict) -> np.ndarray:
        """
        Merge per-instrument residuals into observation space.
        """
        _, merged_residuals = self.batch_processor.merge(
            residuals_dict, mode="eval"
        )

        return merged_residuals.cpu().numpy()

    # =====================================================
    # PLOTTING
    # =====================================================

    def plot(
        self,
        wavelength,
        prediction,
        residuals,
        color="steelblue",
        creds=(0.997, 0.955, 0.683),
        alpha=(0.0, 0.9),
        figsize=(10, 5),
        xlim=(0, 18.2),
        ylim_spectra=(-10, 60),
        ylim_residuals=(-25, 25),
        inset_spectra=None,
        inset_residuals=None,
        rcparams=None,
    ):
        """
        Generalized consistency plot with credible-region bands,
        inset zooms, custom axis limits, and custom legends.

        Args:
            wavelength: (D,) observation wavelengths
            prediction: (B, D) posterior predictive spectra
            residuals: (B, D) residuals (prediction - obs) / sigma
            color: color for the credible-region bands
            creds: credibility levels for shading
            alpha: (min, max) alpha transparency range
            figsize: figure size
            xlim: (xmin, xmax) shared x-axis limits
            ylim_spectra: (ymin, ymax) for the upper panel
            ylim_residuals: (ymin, ymax) for the lower panel
            inset_spectra: dict with keys xlim, ylim, width, height, bbox_to_anchor
            inset_residuals: dict with same keys
            rcparams: optional dict of matplotlib rcParams overrides

        Example:
            fig = evaluator.plot_merged(
                wavelength, prediction, residuals,
                inset_spectra={"xlim": (0.97, 2.2), "ylim": (-0.25, 10)},
                inset_residuals={"xlim": (0.97, 2.2), "ylim": (-10, 20)},
            )
        """

        from lampe.plots import LinearAlphaColormap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        # rc params
        params = rcparams or {
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
        plt.rcParams.update(params)

        # compute levels for colormap
        creds_arr = np.sort(np.asarray(creds))[::-1]
        creds_arr = np.append(creds_arr, 0)
        levels = (creds_arr - creds_arr.min()) / (creds_arr.max() - creds_arr.min())
        levels = (levels[:-1] + levels[1:]) / 2

        cmap = LinearAlphaColormap(color, levels=creds_arr, alpha=alpha)

        fig, (ax1, ax2) = plt.subplots(
            2,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # ---------------------------------
        # upper panel: spectra
        # ---------------------------------

        for q, l in zip(creds_arr[:-1], levels):
            lower, upper = np.quantile(
                prediction, [0.5 - q / 2, 0.5 + q / 2], axis=0
            )
            ax1.fill_between(
                wavelength, lower, upper,
                color=cmap(l), linewidth=0,
            )

        ax1.plot(
            wavelength, self.x_obs.squeeze.cpu().numpy(),
            color="black", linewidth=0.4,
            label=r"$x_{\mathrm{obs}}$",
        )

        # custom legend via legends()
        handles, texts = legends(axes=ax1, alpha=alpha, color=color)
        texts = [r"$x_{\mathrm{obs}}$", r"$p_\phi(f(\theta)|x_{\mathrm{obs}})$"]
        ax1.legend(handles, texts, bbox_to_anchor=(1, 1))

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel(r"Planet flux $F_\nu$ (10$^{-5}$) Jy")
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim_spectra)

        # ---------------------------------
        # inset zoom: spectra
        # ---------------------------------

        if inset_spectra is not None:
            ins = inset_spectra
            ax_ins = inset_axes(
                ax1,
                ins.get("width", 1.8),
                ins.get("height", 0.9),
                loc=2,
                bbox_to_anchor=ins.get("bbox_to_anchor", (0.14, 0.87)),
                bbox_transform=fig.transFigure,
            )

            ax_ins.plot(wavelength, self.x_obs.squeeze.cpu().numpy(), color="black", linewidth=0.4)

            for q, l in zip(creds_arr[:-1], levels):
                lower, upper = np.quantile(
                    prediction, [0.5 - q / 2, 0.5 + q / 2], axis=0
                )
                ax_ins.fill_between(
                    wavelength, lower, upper,
                    color=cmap(l), linewidth=0,
                )

            ax_ins.set_xlim(ins["xlim"])
            ax_ins.set_ylim(ins["ylim"])
            mark_inset(ax1, ax_ins, loc1=4, loc2=3, fc="none", ec="0.7")

        # ---------------------------------
        # lower panel: residuals
        # ---------------------------------

        for q, l in zip(creds_arr[:-1], levels):
            lower, upper = np.quantile(
                residuals, [0.5 - q / 2, 0.5 + q / 2], axis=0
            )
            ax2.fill_between(
                wavelength, lower, upper,
                color=cmap(l), linewidth=0,
            )

        ax2.hlines(
            0, wavelength[0], wavelength[-1],
            color="black", linewidth=0.5,
        )
        ax2.set_xlabel(r"Wavelength ($\mu$m)")
        ax2.set_ylabel(r"Residuals")
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim_residuals)

        # ---------------------------------
        # inset zoom: residuals
        # ---------------------------------

        if inset_residuals is not None:
            ins = inset_residuals
            ax_ins2 = inset_axes(
                ax2,
                ins.get("width", 1.0),
                ins.get("height", 0.4),
                loc=2,
                bbox_to_anchor=ins.get("bbox_to_anchor", (0.22, 0.28)),
                bbox_transform=fig.transFigure,
            )

            ax_ins2.hlines(
                0, wavelength[0], wavelength[-1],
                color="black", linewidth=0.5,
            )

            for q, l in zip(creds_arr[:-1], levels):
                lower, upper = np.quantile(
                    residuals, [0.5 - q / 2, 0.5 + q / 2], axis=0
                )
                ax_ins2.fill_between(
                    wavelength, lower, upper,
                    color=cmap(l), linewidth=0,
                )

            ax_ins2.set_xlim(ins["xlim"])
            ax_ins2.set_ylim(ins["ylim"])
            mark_inset(ax2, ax_ins2, loc1=4, loc2=3, fc="none", ec="0.7")

        return fig
