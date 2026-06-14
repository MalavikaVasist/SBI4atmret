from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import islice
import pandas as pd


@dataclass(frozen=True)
class CoverageResult:
    """
    Container for coverage evaluation outputs.

    Stores:
    - raw SBC/coverage ranks
    - computed empirical coverage curve
    - alpha grid used for plotting
    - optional matplotlib figure
    - optional save path

    Usage:  
    result = evaluator.compute_coverage()
    print(result.calibration_error)
    result.figure.savefig(...)
    """

    # -----------------------------------
    # raw statistics
    # -----------------------------------

    ranks: np.ndarray

    # empirical coverage values
    coverage: np.ndarray

    # credibility levels
    alpha: np.ndarray

    # -----------------------------------
    # optional artifacts
    # -----------------------------------

    figure: Optional[Any] = None

    save_path: Optional[Path] = None


class coverage(BaseEvaluator):
        
    def __call__(self, save_path: Path, *args, **kwargs):
        ranks, coverage, alpha = self.compute_coverage()
        figure = self.plot(coverage)


        if save_path is not None:
            df_cov = pd.DataFrame(coverage) #convert to a dataframe
            df_cov.to_csv(save_path /"coverage.csv",index=False) #save to file        
            figure.savefig(
                save_path/ "coverage.pdf",
                bbox_inches="tight",
            )

        return CoverageResult(
            ranks=ranks,
            coverage=coverage,
            alpha=alpha,
            figure=figure,
            save_path=save_path,
        )

    def compute_coverage(
        self,
    ):

        ranks = []
        with torch.no_grad():
            for batches in islice(zip(*self.test_loaders), 128):
                theta, x = self.batch_processor.prepare_batch(batches, self.test_keys)

                posterior = self.model.flow(x)
                samples = posterior.sample((1024,))
                log_p = posterior.log_prob(theta)
                log_p_samples = posterior.log_prob(samples)

                ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())
        
        ranks = torch.cat(ranks).numpy()
        alpha = np.linspace(0, 1, 100)
        sorted_ranks = np.sort(ranks.flatten())

        coverage = np.array([
            (sorted_ranks > (1 - a)).mean()
            for a in alpha
        ])

        return ranks, coverage, alpha


    def plot(self, coverage):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize = 12)
        ax.set_ylabel(r'Coverage probability', fontsize= 12)
        ax.plot(np.linspace(0,1,100),coverage, color='steelblue', label='') #a[::-1]
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        return fig    