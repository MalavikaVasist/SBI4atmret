

from dataclasses import dataclass

from sbi4atmret.evaluation.EvaluateBase import BaseEvaluator


class Consistencywrapper(BaseEvaluator):

    def compute_posterior_predictive(self, plot=True):

        '''
        find all the simulators from simulator_dict and simulate the 
        posterior predictive for each of them. 
        '''
        return None
    
    def combine_predictive_checks(self, plot=True):

        '''
        combine the posterior predictive checks into a single consistency check based
        on the sort_index. 
        '''
        return None
    

    def plot(self, plot=True):

        '''
        plot and save the consistency check. 
        '''
        return fig



    # def plot





from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .EvaluateBase import BaseEvaluator


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class ConsistencyResult:

    # posterior samples
    posterior_samples: torch.Tensor

    # simulator outputs
    predictive_dict: dict

    # merged spectra
    merged_prediction: np.ndarray

    merged_wavelength: np.ndarray

    # residuals against observation
    residuals: np.ndarray

    # plotting
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

        predictive_dict = (
            self.compute_posterior_predictive(
                posterior_samples
            )
        )

        merged_wavelength, merged_prediction = (
            self.combine_predictive_checks(
                predictive_dict
            )
        )

        # (B, D) - (D,) broadcasts to (B, D)
        residuals = (
            merged_prediction
            - self.x_obs
        )

        figure = None

        if plot:

            figure = self.plot(
                merged_wavelength,
                merged_prediction,
                residuals,
            )

            if save_path is not None:
                figure.savefig(
                    save_path / "posterior_predictive.pdf",
                    bbox_inches="tight",
                )

        return ConsistencyResult(
            posterior_samples=posterior_samples,
            predictive_dict=predictive_dict,
            merged_prediction=merged_prediction,
            merged_wavelength=merged_wavelength,
            residuals=residuals,
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
        savepath : 
    ) -> dict:

        """
        Simulate spectra from all simulators.

        Returns a dict matching the pipe's batch_dict format:
            {sim_name: (theta, x)}
        where theta is (B, D_inst) and x is (B, D_spec).
        """

        # split_theta expects (B, D) — add batch dim if needed
        if posterior_samples.dim() == 1:
            posterior_samples = posterior_samples.unsqueeze(0)

        theta_dict = self.pipe.split_theta(posterior_samples)

        predictive = {}

        for sim_name, simulator in (
            self.simulator_dict.items()
        ):

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

            predictive[sim_name] = (
                theta_dict[sim_name],  # (B, D_inst)
                x,                     # (B, D_spec)
            )

        

        return predictive

    # =====================================================
    # MERGING
    # =====================================================

    def combine_predictive_checks(self, predictive_dict: dict):
        """
        Apply pipe transforms and merge into observation space.
        Uses batch_processor in eval mode (skips modify_theta).
        """
        _, x = self.batch_processor.prepare_batch(
            predictive_dict,
            mode="eval",
            add_noise=True,
        )

        merged_wavelength = self.observation.full_wavelength

        return merged_wavelength, x.cpu().numpy()


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




    
 # def evaluate_posterior_predictive(self, theta_posterior, simulators_dict):
    #     """
    #     Generate posterior predictive samples from posterior theta samples.
        
    #     Typical usage during evaluation:
    #         pipe = MiriGeminiHSTcloudfreePipe(config)
    #         # ... train network ...
            
    #         theta_posterior = posterior.sample((10000,))  # 10k posterior samples
    #         spec_predictive = pipe.evaluate_posterior_predictive(
    #             theta_posterior, 
    #             {"miri": miri_sim, "gemini": gemini_sim, "hst": hst_sim}
    #         )
        
    #     Args:
    #         theta_posterior: [B, D_global] posterior samples
    #         simulators_dict: Dict with keys "cloudfree_miri", "cloudfree_gemini", "cloudfree_hst" containing simulator functions
            
    #     Returns:
    #         Dict with keys "cloudfree_miri", "cloudfree_gemini", "cloudfree_hst" containing simulated spectra [B, D]
    #     """
    #     # Split theta back to simulator-specific parameters
    #     theta_dict = self.pipe.split_theta(theta_posterior)
        
    #     # Run simulators independently
    #     specs = {
    #         inst: simulators_dict[inst](theta_dict[inst])
    #         for inst in theta_dict.keys()
    #     }
        
    #     return specs
