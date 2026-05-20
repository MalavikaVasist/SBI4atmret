
from pathlib import Path
from typing import Optional
import logging
from sbi4atmret.utils.checkpoint import load_model_checkpoint

# Setup logging
logger = logging.getLogger(__name__)


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
        self.net = model.estimator
        self.context = context
        self.config = config
        self.device = context.device
        self.test_lists = self.context.test_lists

        # Extract from context
        self.runtime = context.runtime
        self.test_lists = context.test_lists
        self.checkpoint_path = Path(context.checkpoint_path) if context.checkpoint_path else None

        # Setup save directories
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.savefolder = self.checkpoint_path.parent.parent
        else:
            self.savefolder = Path.cwd()

        self.eval_dir = self.savefolder / "evaluations"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Load and setup model
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load_model_state(self.net, self.checkpoint_path)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
        
        self.net.to(self.device)

        # Build posterior
        self.posterior = model.flow()

    def load_model_state(self, net, checkpoint_path):
        """Load model state from checkpoint."""
        try:
            checkpoint = load_model_checkpoint(checkpoint_path)
            net.load_state_dict(checkpoint['estimator'])
            logger.info("Model state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model state: {e}")

    def build_posterior(self):
        """Build posterior from the model."""
        posterior = self.net.inference.build_posterior(self.net.estimator)
        return posterior

    def run_all(self):
        """Run all evaluation methods."""
        self.run_coverage()
        self.run_sbc()
        self.run_corner()
        self.run_pt_profile()
        self.run_posterior_predictive()

    def run_corner(self):
        """Run corner plot evaluation."""
        from .Plots import make_corner_plot
        make_corner_plot(
            posterior=self.posterior,
            dataset=self.test_lists,
            config=self.config,
            save_path=self.eval_dir / "corner.png",
        )

    def run_coverage(self):
        """Run coverage evaluation."""
        from .coverage import compute_coverage
        self.coverage = compute_coverage(
            posterior=self.posterior,
            dataset=self.test_lists,
            config=self.config,
            save_path=self.eval_dir / "coverage.png",
        )

    def run_sbc(self):
        """Run SBC (Simulation-Based Calibration) evaluation."""
        from .consistency import compute_sbc
        self.consistency = compute_sbc(
            posterior=self.posterior,
            dataset=self.test_lists,
            config=self.config,
            save_path=self.eval_dir / "sbc.png",
        )

    def run_pt_profile(self):
        """Run PT profile evaluation."""
        from .PT_profile import compute_pt_profile
        self.PTprofile = compute_pt_profile(
            posterior=self.posterior,
            dataset=self.test_lists,
            config=self.config,
            save_path=self.eval_dir / "pt_profile.png",
        )

    def run_posterior_predictive(self):
        """Run posterior predictive evaluation."""
        from .Plots import compute_posterior_predictive
        self.IS = compute_posterior_predictive(
            posterior=self.posterior,
            dataset=self.test_lists,
            config=self.config,
            save_path=self.eval_dir / "posterior_predictive.png",
        )

    def perform_evaluations(self):
        """Alias for run_all."""
        self.run_all()