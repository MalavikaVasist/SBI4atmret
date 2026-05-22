

# from sbi4atmret.evaluation.EvaluateBase import BaseEvaluator


# class CosnistencyEvaluator(BaseEvaluator):
#     def __init__():
#         super().__init__()


#     def compute_consistency(self, plot=True):





    
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
    
