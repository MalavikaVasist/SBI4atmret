

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


@dataclass(frozen=True)
class ConsistencyEvaluator:

    posterior_samples: torch.Tensor

    predictive_samples: dict

    wavelengths: dict

    residuals: dict

    merged_prediction: Optional[np.ndarray]

    figure: Optional[Any] = None






    
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
    
