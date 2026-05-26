
from sbi4atmret.utils.checkpoint import load_checkpoint, load_model_state
from sbi4atmret.runtime.batch_processor import BatchProcessor
import torch
import pandas as pd
from tqdm import tqdm


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

        # Build posterior
        self.posterior = self.build_posterior()

        # Build posterior
        self.batch_processor = BatchProcessor(
                            dataset=self.dataset,
                            pipe=self.pipe,
                            noise=self.noise,
                            device=self.device,
                        )
        
        ## sampling from posterior
        self.x_obs = self.observation.full_observation
        self.theta = self.sampling_from_post(
                                        torch.from_numpy(self.x_obs).unsqueeze(0).float().to(self.device), 
                                        self.eval_dir/'theta.csv', 
                                        only_returning = False) 

    def build_posterior(self):
        """Build posterior from the model."""
        posterior = self.model.flow().to(self.device)
        return posterior
    
    def sampling_from_post(self, x, name, only_returning = True):
    
        if not only_returning: 
            with torch.no_grad():
                theta = torch.cat([
                    self.model.flow(x).sample((2**14,)).cpu()
                    for _ in tqdm(range(2**6))
                ])
                theta = theta.squeeze()
            ##Saving to file
            theta_numpy = theta.double().numpy() #convert to Numpy array
            df_theta = pd.DataFrame(theta_numpy) #convert to a dataframe
            df_theta.to_csv( name ,index=False) #save to file
            return theta
        
        #Then, to reload:
        df_theta = pd.read_csv(name)
        theta = df_theta.values
        return torch.from_numpy(theta)


    def run_all(self):
        """Run all evaluation methods."""
        self.run_coverage()
        self.run_sbc()
        self.run_corner()
        self.run_pt_profile()
        self.run_posterior_predictive()
        

    def perform_evaluations(self):
        """Alias for run_all."""
        self.run_all()


    # def run_sbc(self):
    #     """Run SBC (Simulation-Based Calibration) evaluation."""
    #     from .consistency import compute_sbc
    #     self.consistency = compute_sbc(
    #         posterior=self.posterior,
    #         dataset=self.test_lists,
    #         config=self.config,
    #         save_path=self.eval_dir / "sbc.pdf",
    #     )