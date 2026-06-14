from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .EvaluateBase import BaseEvaluator
from sbi4atmret.utils.general import instrument_from_simname


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class ConsistencyResult:

    posterior_samples: torch.Tensor

    predictive_dict: dict

    residuals: dict

    merged_prediction: np.ndarray

    merged_wavelength: np.ndarray

    merged_residuals: np.ndarray

    figure: Optional[Any] = None


# =========================================================
# EVALUATOR
# =========================================================

class ConsistencyEvaluator(BaseEvaluator):

    """
    Posterior predictive consistency checks.

    Runs posterior samples through all simulators,
    merges predictions into observation space,
    and compares against observed spectra.
    """

    # =====================================================
    # MAIN API
    # =====================================================

    def run(
        self,
        n_posterior_samples: int = 512,
        plot: bool = True,
        save_path: Optional[Path] = None,
    ) -> ConsistencyResult:

        posterior_samples = self.consistency_samples(
            n_samples=n_posterior_samples,
        )

        predictive_dict = self.compute_posterior_predictive(
            posterior_samples,
            savepath=save_path,
        )

        residuals_dict = self.compute_residuals(predictive_dict)

        merged_wavelength, merged_prediction = self.combine_predictives(
            predictive_dict
        )

        merged_residuals = self.combine_residuals(residuals_dict)

        figure = None

        if plot:

            figure = self.plot(
                merged_wavelength,
                merged_prediction,
                merged_residuals,
            )

            if save_path is not None:
                figure.savefig(
                    save_path / "posterior_predictive.pdf",
                    bbox_inches="tight",
                )

        return ConsistencyResult(
            posterior_samples=posterior_samples,
            predictive_dict=predictive_dict,
            residuals = residuals, 
            merged_prediction=merged_prediction,
            merged_wavelength=merged_wavelength,
            merged_residuals=merged_residuals,
            figure=figure,
        )

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
        savepath: Optional[Path] = None,
    ) -> dict:

        """
        Simulate spectra from all simulators.

        Returns a dict matching the pipe's batch_dict format:
            {sim_name: (theta, x)}
        where theta is (B, D_inst) and x is (B, D_spec).

        The returned dict has pipe spectral transforms and noise applied.
        """

        # split_theta expects (B, D) — add batch dim if needed
        if posterior_samples.dim() == 1:
            posterior_samples = posterior_samples.unsqueeze(0)

        theta_dict = self.pipe.split_theta(posterior_samples)

        predictive_dict = {}

        for sim_name, simulator in self.simulator_dict.items():

            spectra = []

            for i in tqdm(
                range(theta_dict[sim_name].shape[0]),
                desc=f"{sim_name}",
            ):

                theta_i = theta_dict[sim_name][i].numpy()
                output = simulator(theta_i)
                spectra.append(
                    torch.from_numpy(output.spectrum).float()
                )

            x = torch.stack(spectra)  # (B, D_spec)

            predictive_dict[sim_name] = (
                theta_dict[sim_name],  # (B, D_inst)
                x,                     # (B, D_spec)
            )

        # Apply spectral transforms (no modify_theta) and noise
        predictive_dict = self.batch_processor.process(
            predictive_dict, mode="eval", add_noise=True
        )

        if savepath is not None:
            savepath.mkdir(parents=True, exist_ok=True)
            torch.save(predictive_dict, savepath / "predictive_dict.pt")

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
        figsize=(10, 6),
    ):

        """
        Generalized consistency plot with credible-region bands.

        Upper panel: posterior predictive spectra vs observation.
        Lower panel: residuals.

        Args:
            wavelength: (D,) observation wavelengths
            prediction: (B, D) posterior predictive spectra
            residuals: (B, D) residuals (prediction - obs) / sigma
            color: color for the credible-region bands
            creds: credibility levels for shading
            alpha: (min, max) alpha transparency range
            figsize: figure size
        """

        from lampe.plots import LinearAlphaColormap

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
            sharex=True,
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
            wavelength, self.x_obs,
            color="black", linewidth=0.4,
            label=r"$x_{\mathrm{obs}}$",
        )

        # build legend with credible-region patches
        import matplotlib.patches as mpatches

        handles, texts = ax1.get_legend_handles_labels()
        for q, l in zip(creds_arr[:-1], levels):
            handles.append(mpatches.Patch(color=cmap(l), linewidth=0))
            texts.append(r"${:.1f}\,\%$ credible region".format(q * 100))

        ax1.legend(handles, texts, bbox_to_anchor=(1, 1))
        ax1.set_ylabel(r"Planet flux $F_\nu$ ($10^{-5}$ Jy)")
        plt.setp(ax1.get_xticklabels(), visible=False)

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

        ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax2.set_xlabel(r"Wavelength ($\mu$m)")
        ax2.set_ylabel(r"Residuals")

        fig.tight_layout()

        return fig
