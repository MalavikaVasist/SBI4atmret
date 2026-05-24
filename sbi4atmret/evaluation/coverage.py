
from itertools import islice
from EvaluateBase import BaseEvaluator

import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class CoverageEvaluator(BaseEvaluator):

    def compute_coverage(self, plot=True, save_path=None): 
        ranks = []
        with torch.no_grad():
            for batches in islice(zip(*self.test_loaders), 128):
                theta, x = self.batch_processor.prepare_batch(batches, self.test_keys)

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
            image = self.plot(a)
            if save_path is not None:
                image.savefig(save_path, bbox_inches='tight')
                
        return image


    def plot(self, a):
        cov_fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize = 12)
        ax.set_ylabel(r'Coverage probability', fontsize= 12)
        ax.plot(np.linspace(0,1,100),a, color='steelblue', label='') #a[::-1]
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)
        return cov_fig    