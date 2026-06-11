

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
    predictive_samples: dict

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

        residuals = (
            merged_prediction
            - self.x_obs.cpu().numpy()
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
                    save_path/ "posterior_predictive.pdf",
                    bbox_inches="tight",
                )

        return ConsistencyResult(
            posterior_samples=posterior_samples,
            predictive_samples=predictive_samples,
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



    def compute_posterior_predictive(
        self,
        posterior_samples: torch.Tensor,
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
    ):

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
        )

        # ---------------------------------
        # spectrum
        # ---------------------------------

        axes[0].plot(
            wavelength,
            prediction,
            label="Posterior predictive",
        )

        axes[0].scatter(
            wavelength,
            self.x_obs,
            s=10,
            label="Observation",
        )

        axes[0].set_ylabel(
            "Flux"
        )

        axes[0].legend()

        # ---------------------------------
        # residuals
        # ---------------------------------

        axes[1].plot(
            wavelength,
            residuals,
        )

        axes[1].axhline(
            0,
            linestyle="--",
        )

        axes[1].set_xlabel(
            "Wavelength [micron]"
        )

        axes[1].set_ylabel(
            "Residual"
        )

        return fig




    
