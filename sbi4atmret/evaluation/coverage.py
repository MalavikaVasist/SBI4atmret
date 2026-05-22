
from itertools import islice
from logging import config
from turtle import pd
from EvaluateBase import BaseEvaluator

from sbi4atmret.torchutils.general import _to_device
import torch
import numpy as np
from matplotlib import pyplot as plt


class CoverageEvaluator(BaseEvaluator):

    def __init__():
        super().__init__()

    def compute_coverage(self, plot=True): 
        ranks = []
        with torch.no_grad():
            for batches in islice(zip(*self.test_loaders), 128):
                theta, x = self.batch_processor.prepare_batch(batches, self.test_keys)
                # theta, x  = _to_device(theta), _to_device(x)
                theta, x = theta.to(self.device), x.to(self.device)


                posterior = self.estimator.flow(x)
                samples = posterior.sample((1024,))
                log_p = posterior.log_prob(theta)
                log_p_samples = posterior.log_prob(samples)

                ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

        ranks = torch.cat(ranks)   
        ranks_numpy = ranks.double().numpy() #convert to Numpy array
        df_ranks = pd.DataFrame(ranks_numpy) #convert to a dataframe
        df_ranks.to_csv(self.eval_dir /"ranks.csv",index=False) #save to file

        df_ranks = pd.read_csv(self.eval_dir/"ranks.csv")
        ranks = df_ranks.values

        # Coverage
        a=[]
        r = np.sort(np.asarray(ranks))

        for alpha in np.linspace(0,1,100):
            a.append((r > (1-alpha)).mean())

        if plot: 
            self.plot_coverage(a)

    def plot_coverage(self, a):
        cov_fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize = 12)
        ax.set_ylabel(r'Coverage probability', fontsize= 12)
        ax.plot(np.linspace(0,1,100),a, color='steelblue', label='upper right') #a[::-1]
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        cov_fig.savefig(self.eval_dir / 'coverage.pdf') 
        return cov_fig    