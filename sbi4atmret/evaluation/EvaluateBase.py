
import logging
from sbi4atmret.utils.checkpoint import load_checkpoint, load_model_state
from sbi4atmret.runtime.batch_processor import BatchProcessor
from utils import to_device

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
        self.context = context
        self.config = config

        ## domain components
        self.domain = context.runtime.domain

        self.simulator = self.domain.simulators
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


        self.batch_processor = BatchProcessor(
                            dataset=self.dataset,
                            pipe=self.pipe,
                            noise=self.noise,
                            device=self.device,
                        )

    def build_posterior(self):
        """Build posterior from the model."""
        posterior = self.model.flow().to(self.device)
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
            self.batch_processor,
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