
from sbi4atmret.utils.checkpoint import load_checkpoint, load_model_state
from sbi4atmret.runtime.batch_processor import BatchProcessor
import torch
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from .coverage import CoverageResult, compute_coverage, plot_coverage
from .consistency import ConsistencyEvaluator


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
        

    def perform_evaluations(self):
        """Alias for run_all."""
        self.run_all()



    def run_coverage():
        coverage_result = coverage(self.eval_dir)


    def run_consistency():
        compute_consistency()


