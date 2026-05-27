

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

        predictive_samples = (
            self.compute_posterior_predictive(
                posterior_samples
            )
        )

        merged_wavelength, merged_prediction = (
            self.combine_predictive_checks(
                predictive_samples
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
                    save_path,
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


    def evaluate_posterior_predictive(self, theta_posterior):
        """
        Generate posterior predictive samples from posterior theta samples.
        
        Typical usage during evaluation:
            pipe = MiriGeminiHSTcloudfreePipe(config)
            # ... train network ...
            
            theta_posterior = posterior.sample((10000,))  # 10k posterior samples
            spec_predictive = pipe.evaluate_posterior_predictive(
                theta_posterior, 
                {"miri": miri_sim, "gemini": gemini_sim, "hst": hst_sim}
            )
        
        Args:
            theta_posterior: [B, D_global] posterior samples
            simulators_dict: Dict with keys "miri", "gemini", "hst" containing simulator functions
            
        Returns:
            Dict with keys "miri", "gemini", "hst" containing simulated spectra [B, D]
        """
        # Split theta back to simulator-specific parameters
        thetam, thetag, thetah = self.domain.pipe.split_theta(theta_posterior)
        
        # Run simulators independently
        specs = {
            "miri": self.domain.simulator_dict["miri"](thetam),
            "gemini": self.domain.simulator_dict["gemini"](thetag),
            "hst": self.domain.simulator_dict["hst"](thetah)
        }
        
        return specs

    def compute_posterior_predictive(
        self,
        posterior_samples: torch.Tensor,
    ) -> dict:

        """
        Simulate spectra from all simulators.
        """

        predictive = {}

        for sim_name, simulator in (
            self.simulator_dict.items()
        ):

            outputs = []

            for theta in tqdm(
                posterior_samples,
                desc=f"{sim_name}",
            ):

                output = simulator(
                    theta.numpy()
                )

                outputs.append(output)

            predictive[sim_name] = outputs

        return predictive

    # =====================================================
    # MERGING
    # =====================================================

    def combine_predictive_checks(
        self,
        predictive_samples: dict,
    ):

        """
        Merge all simulator outputs into
        observation-space ordering.
        """

        merged_spectra = []
        merged_wavelengths = []

        for sim_name, outputs in (
            predictive_samples.items()
        ):

            spectra = np.stack([
                o.spectrum
                for o in outputs
            ])

            wavelength = outputs[0].wavelength

            merged_spectra.append(
                spectra.mean(axis=0)
            )

            merged_wavelengths.append(
                wavelength
            )

        merged_prediction = np.concatenate(
            merged_spectra
        )

        merged_wavelength = np.concatenate(
            merged_wavelengths
        )

        # restore observation ordering
        unsort = self.domain.unsort_index

        merged_prediction = (
            merged_prediction[unsort]
        )

        merged_wavelength = (
            merged_wavelength[unsort]
        )

        return (
            merged_wavelength,
            merged_prediction,
        )

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




    
#     def consistencyplot_MIRI(self):
#             # wlen = obs_wlen_hst
#         self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

#         fig = MIRI_consistency( self.theta[:512], 
#                                 simulator_miri_cloudy = None,
#                                 simulator_miri_cloudfree = simulator_miri_cloudfree,
#                                 savepath_plots = self.savepath_plots,
#                                 cloud = 'cloudfree', 
#                                 obs_miri = obs_miri, 
#                                 obs_wlen_miri = obs_wlen_miri, 
#                                 sigmaM = sigmaM,
#                                 only_returning = False,
#                                 p = None).fig
#         return fig

#     def consistencyplot_Gemini(self):
#         # wlen = obs_wlen_hst
#         self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

#         fig = Gemini_consistency(  self.theta[:512], 
#                                 simulator_hst_cloudy = None,
#                                 simulator_hst_cloudfree = simulator_hst_cloudfree,
#                                 mode = 'MIRI + HST+ Gemini', 
#                                 savepath_plots = self.savepath_plots,
#                                 cloud = 'cloudfree',
#                                 obs_gemini = obs_gemini, 
#                                 obs_wlen_gemini = obs_wlen_gemini,
#                                 sigmaG = sigmaG,  
#                                 only_returning = False,
#                                 p = None).fig
#         return fig
    
#     def consistencyplot_HST(self):
#         # wlen = obs_wlen_hst
#         self.theta = self.sampling_from_post(torch.from_numpy(self.x_star).float().cuda(), self.savepath_plots/'theta.csv', only_returning = True)

#         fig = HST_consistency(  self.theta[:512], 
#                                 simulator_hst_cloudy = simulator_hst_cloudfree,
#                                 simulator_hst_cloudfree = simulator_hst_cloudfree,
#                                 mode = 'MIRI + HST+ Gemini', 
#                                 savepath_plots = self.savepath_plots,
#                                 cloud = 'cloudfree',
#                                 obs_hst = obs_hst, 
#                                 obs_wlen_hst = obs_wlen_hst,
#                                 sigmaH = sigmaH, 
#                                 only_returning = False,
#                                 p = None, 
#                                 ).fig
        

                
#         return fig
    
