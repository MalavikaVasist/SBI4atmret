
from pathlib import Path
from typing import Union, Optional
from utils.checkpoint import load_model_checkpoint

class BaseEvaluator:

    def __init__(
        self,
        model,
        dataset,
        config,
        checkpoint_path,
        device="cuda",
    ):

        self.net = model.estimator
        self.dataset = dataset
        self.config = config
        self.device = device

        self.checkpoint_path = Path(checkpoint_path)

        self.savefolder = self.checkpoint_path.parent.parent

        self.eval_dir = self.savefolder / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.load_model_state(self.net,
                            self.checkpoint_path,
                            ) 
        self.net.to(self.device)

        self.posterior = self.build_posterior()



    def build_posterior(self):

        posterior = self.net.inference.build_posterior(
            self.net.estimator
        )

        return posterior

    
    def run_all(self):

        self.run_coverage()

        self.run_sbc()

        self.run_corner()

        self.run_pt_profile()

        self.run_posterior_predictive()

    
    def run_corner(self):

        from .corner import make_corner_plot
        make_corner_plot(
            posterior=self.posterior,
            dataset=self.dataset,
            config=self.config,
            save_path=self.eval_dir / "corner.png",
        )

    def run_coverage(self):

        from .coverage import compute_coverage
        self.coverage = compute_coverage(
            posterior=self.posterior,
            dataset=self.dataset,
            config=self.config,
            save_path=self.eval_dir / "coverage.png",
        )

    def run_sbc(self):

        from .sbc import compute_sbc
        self.consistency = compute_sbc(
            posterior=self.posterior,
            dataset=self.dataset,
            config=self.config,
            save_path=self.eval_dir / "sbc.png",
        )

    def run_pt_profile(self):
        
        from .pt_profile import compute_pt_profile
        self.PTprofile = compute_pt_profile(
            posterior=self.posterior,
            dataset=self.dataset,
            config=self.config,
            save_path=self.eval_dir / "pt_profile.png",
        )

    def run_posterior_predictive(self):

        from .posterior_predictive import compute_posterior_predictive
        self.IS = compute_posterior_predictive(
            posterior=self.posterior,
            dataset=self.dataset,
            config=self.config,
            save_path=self.eval_dir / "posterior_predictive.png",
        )


    def perform_evaluations(self):

        self.run_all()