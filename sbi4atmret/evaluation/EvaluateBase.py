
from sbi4atmret.utils.checkpoint import load_checkpoint, load_model_state
from sbi4atmret.runtime.batch_processor import BatchProcessor
import torch
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from .coverage import CoverageResult, compute_coverage, plot_coverage
from .consistency import ConsistencyEvaluator, ConsistencyResult


class BaseEvaluator:

    def __init__(
        self,
        model,
        context,
        config,
    ):
        """
        Initialize the BaseEvaluator.
        
        Args:
            model: The model with estimator and flow methods
            context: EvaluationContext with runtime, test_lists, checkpoint_path, device
            config: Configuration object
            dataset: Optional dataset (for backward compatibility)
        """

        self.model = model
        self.context = context
        self.config = config

        ## domain components
        self.domain = context.runtime.domain

        self.simulator_dict = self.domain.simulator_dict
        self.observation = self.domain.observation
        self.pipe = self.domain.pipe
        self.noise = self.domain.noise
        
        self.checkpoint_path = context.runtime.checkpoint_path
        self.device = context.runtime.device

        ## dataset components
        self.test_keys, self.test_loaders = context.test_lists

        # Setup save directories
        self.savefolder = self.checkpoint_path.parent.parent
        self.eval_dir = self.savefolder / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Load and setup model
        self.net = self.model.estimator
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        load_model_state(self.net, checkpoint)
        self.net.to(self.device)

        self.batch_processor = BatchProcessor(
                            pipe=self.pipe,
                            noise=self.noise,
                            device=self.device,
                        )
        
        ## sampling from posterior
        self.x_obs = torch.from_numpy(self.observation.full_observation).unsqueeze(0).float().to(self.device)
        self.posterior = self.build_posterior(self.x_obs)
        self.theta = self.sampling_from_post(
                                            filename = self.eval_dir/'theta.csv',
                                            posterior= posterior, 
                                            only_returning = False) 


    def build_posterior(self, x):
        """Build posterior from the model."""
        posterior = self.net.flow_forward(x).to(self.device)
        return posterior
    
    def sampling_from_post(self, filename= None, only_returning = True, posterior= None):
    
        if not only_returning: 
            with torch.no_grad():
                theta = torch.cat([
                    posterior.sample((2**14,)).cpu()
                    for _ in tqdm(range(2**6))
                ])
                theta = theta.squeeze()

            ##Saving to file
            theta_numpy = theta.double().numpy() #convert to Numpy array
            df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
            df_theta.to_csv( filename ,index=False) #save to file
            return theta
        
        #Then, to reload:
        df_theta = pd.read_csv(filename)
        theta = df_theta.values
        return torch.from_numpy(theta)


    def run_all(self):
        """Run all evaluation methods."""
        self.run_coverage()
        self.run_corner()
        self.run_pt_profile()
        self.run_posterior_predictive()
        self.run_consistency()
        self.IS_corner()
        

    def perform_evaluations(self):
        """Alias for run_all."""
        self.run_all()



    def run_coverage(self):
        """Compute coverage, save CSV and plot to eval_dir."""
        import numpy as np

        self.coverage_path = self.eval_dir / "coverage" 
        self.coverage_path.mkdir(exist_ok=True, parents=True)

        csv_path = self.coverage_path / "coverage.csv"
        alpha = np.linspace(0, 1, 100)

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            coverage = df["coverage"].values
            ranks = None
        else:
            ranks, coverage, alpha = compute_coverage(
                net=self.net,
                batch_processor=self.batch_processor,
                test_loaders=self.test_loaders,
                test_keys=self.test_keys,
            )
            df = pd.DataFrame({"alpha": alpha, "coverage": coverage})
            df.to_csv(csv_path, index=False)

        figure = plot_coverage(coverage, alpha)
        figure.savefig(
            self.coverage_path / "coverage.pdf",
            bbox_inches="tight",
        )

        return CoverageResult(
            ranks=ranks,
            coverage=coverage,
            alpha=alpha,
            figure=figure,
            save_path=self.eval_dir,
        )

    def run_consistency(self):
        """Run posterior predictive consistency check.
        
        Checks for saved predictive_dict.pt and residuals_dict.pt.
        If found, loads and skips to combine + plot.
        If not, generates from scratch then saves.
        """
        consistency_eval = ConsistencyEvaluator.__new__(ConsistencyEvaluator)
        consistency_eval.__dict__.update(self.__dict__)

        self.consistency_path = self.eval_dir / "consistency" 
        self.consistency_path.mkdir(exist_ok=True, parents=True)


        posterior_samples_path = self.consistency_path/ "posterior_samples.pt"
        predictive_path = self.consistency_path / "predictive_dict.pt"
        residuals_path = self.consistency_path / "residuals_dict.pt"

        if predictive_path.exists() and residuals_path.exists():
            posterior_samples = torch.load(posterior_samples_path, map_location=self.device)
            predictive_dict = torch.load(predictive_path, map_location=self.device)
            residuals_dict = torch.load(residuals_path, map_location=self.device)
        else:
            posterior_samples = consistency_eval.consistency_samples()
            torch.save(posterior_samples, posterior_samples_path)

            predictive_dict = consistency_eval.compute_posterior_predictive(posterior_samples)
            torch.save(predictive_dict, predictive_path)

            residuals_dict = consistency_eval.compute_residuals(predictive_dict)
            torch.save(residuals_dict, residuals_path)

        # combine and plot
        merged_wavelength, merged_prediction = consistency_eval.combine_predictives(
            predictive_dict
        )
        merged_residuals = consistency_eval.combine_residuals(residuals_dict)

        figure = consistency_eval.plot(
            merged_wavelength,
            merged_prediction,
            merged_residuals,
        )

        figure.savefig(
            self.consistency_path / "posterior_predictive.pdf",
            bbox_inches="tight",
        )

        return ConsistencyResult(
            posterior_samples=posterior_samples,
            predictive_dict=predictive_dict,
            residuals_dict=residuals_dict,
            merged_prediction=merged_prediction,
            merged_wavelength=merged_wavelength,
            merged_residuals=merged_residuals,
            figure=figure,
        )






