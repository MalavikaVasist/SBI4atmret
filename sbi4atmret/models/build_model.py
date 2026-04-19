"""
Method to build (i.e., instantiate) a model from a checkpoint file or
from a configuration dictionary.
"""

from pathlib import Path
from typing import Any

from sbi4atmret.MLmodel import estimator
from sbi4atmret.config.configs import EstimatorConfig
from sbi4atmret.utils import config
from sbi4atmret.MLmodel.flows import build_flow
from sbi4atmret.MLmodel.embedding import build_embedding
from scripts.train_general import load_callable

from lampe.nn import ResMLP
from lampe.distributions import NPE
import torch
import torch.nn as nn

from MLmodel import estimator
'''
Here load the flows and embeddings and combine them to build the model.

'''

def build_embedding(embedding_config):
    embedding_type = embedding_config.type 
    embedding = load_callable("sbi4atmret.MLmodel.embedding", embedding_type)
    return embedding(embedding_config)

def build_flow(flow_config):
    flow_type = flow_config.type 
    flow = load_callable("sbi4atmret.MLmodel.flows", flow_type)
    return flow
                    
def build_model(config):
    flow_config, embedding_config, prior_config = EstimatorConfig(**config["flow"], **config["embedding"], **config["Prior"])
    embedding = build_embedding(embedding_config)
    flow = build_flow(flow_config, prior_config, embedding_config) 
    model = estimator(flow, embedding)   
    return model
   


# def _build_flow_estimator(self):
#         """Build a flow-based estimator from the selected config."""
#         # For backward compatibility, pass dict to build_model_estimator
#         config_dict = self.selected_config.model_dump() if self.selected_config else self.config.model_dump()
#         return build_model_estimator(config_dict)

#     def _build_estimator(self, model_config):
#         """Build estimator from ModelConfig."""
#         if model_config is None:
#             return build_model_estimator(self.selected_config.model_dump() if self.selected_config else self.config.model_dump())

#         # For backward compatibility with build_model_estimator which expects dict
#         config_dict = self.selected_config.model_dump() if self.selected_config else self.config.model_dump()
#         return build_model_estimator(config_dict)

#     def setup_estimator(self, i: int = 0):
#         """Set up the estimator for config index i."""
#         config = self._get_active_config(i)
#         model_config = config.ML_model_config
#         self.estimator = self._build_estimator(model_config)
#         if hasattr(self.estimator, 'cuda'):
#             self.estimator = self.estimator.cuda()
#         return self.estimator

        

# class NPEWithEmbedding_sepEmb(nn.Module):
#     def __init__(self, 
#                  hf_miri= [2,3,5], 
#                  hf_inst= [3,5,7], 
#                  instrument= 'HST',
#                 Resinp = 129, 
#                 ResM = 1298, 
#                  hidden_features = [512],
#                  no_of_hidden_features= 5,
#                  emb_miri_output = 64 ,
#                  emb_inst_output = 8 ,
#                  no_of_params = 29,
#                  transforms = 5, 
#                 signal = 16,
#                 LOWER = None, 
#                 UPPER = None, 

#                 ):
        
#         super().__init__()
        
#         self.hf_miri = hf_miri
#         self.hf_inst = hf_inst
#         self.instrument = instrument
#         self.hidden_features = hidden_features
#         self.no_of_hidden_features= no_of_hidden_features
#         self.emb_miri_output = emb_miri_output
#         self.emb_inst_output = emb_inst_output
#         self.no_of_params = no_of_params
#         self.transforms = transforms
#         self.signal = signal
#         self.LOWER = LOWER
#         self.UPPER = UPPER
#         self.Resinp = Resinp
#         self.ResM = ResM

#         ## building the hidden_features array
#         x = torch.Tensor([[512] , [256] , [128]]).int()
#         c = torch.Tensor([]).int()
#         d = torch.Tensor([]).int()
#         for i in range(3):
#             c = torch.hstack( (c, x[i].repeat(self.hf_miri[i])) )
#             d = torch.hstack( (d, x[i].repeat(self.hf_inst[i])) )
                    
#         if self.instrument == 'HST':
#             self.Resinp = 129
#         elif self.instrument == 'Gemini':
#             self.Resinp = 305
#         elif self.instrument == 'HST+Gemini':
#             self.Resinp = 434
            
#         self.embedding_miri = nn.Sequential(
#             SoftClip(100.0),
#             ResMLP(
#                 self.ResM , self.emb_miri_output , #1298 + 129 / 305
#                 hidden_features= c.tolist(), #[512] * 3 + [256] * 5 + [128] * 7, #
#                 activation=nn.ELU,
#             ),
#         )

        
#         self.embedding_inst = nn.Sequential(
#         SoftClip(100.0),
#         ResMLP(
#             self.Resinp , self.emb_inst_output, #1298 + 129 / 305
#             hidden_features=  d.tolist(), #[512] * 3 + [256] * 5 + [128] * 7, #
#             activation=nn.ELU,
#         ),
#     )
        
#         l, u = torch.tensor(self.LOWER), torch.tensor(self.UPPER)

#         self.npe = NPE(
#             self.no_of_params, self.emb_inst_output + self.emb_miri_output ,
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms= self.transforms,
#             build=NAF,
#             signal = self.signal,
#             hidden_features= self.hidden_features * self.no_of_hidden_features,
#             activation='ELU',
#         )

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         return self.npe(theta, torch.hstack(( self.embedding_inst(x[:,:-self.ResM]), self.embedding_miri(x[:,-self.ResM:]) )))
        

#     def flow(self, x: Tensor):  # -> Distribution
#         return self.npe.flow(torch.hstack(( self.embedding_inst(x[:,:-self.ResM]), self.embedding_miri(x[:,-self.ResM:]))))
    
        
# class NPEWithEmbedding_oneEmb(nn.Module):
#     def __init__(self,
#                  hf = [2,3,5] ,
#                  instrument= 'MIRI', 
#                  hidden_features = [512],
#                  no_of_hidden_features= 5,
#                  emb_output = 64,
#                  no_of_params = 31,
#                  transforms = 3, 
#                  signal = 8,
#                  LOWER = None, 
#                  UPPER = None, 
#                  embedding = True,
#                 ):
#         super().__init__()

#         self.hf = hf
#         self.instrument = instrument
#         self.hidden_features = hidden_features
#         self.no_of_hidden_features= no_of_hidden_features
#         self.emb_output = emb_output
#         self.no_of_params = no_of_params
#         self.transforms = transforms
#         self.signal = signal
#         self.LOWER = LOWER
#         self.UPPER = UPPER
            
#         ## building the hidden_features array
#         x = torch.Tensor([[512] , [256] , [128]]).int()
#         d = torch.Tensor([]).int()
#         for i in range(3):
#             d = torch.hstack( (d, x[i].repeat(self.hf[i])) )
                    
#         if self.instrument == 'HST':
#             Resinp = 129
#         elif self.instrument == 'Gemini':
#             Resinp = 305
#         elif self.instrument == 'MIRI':
#             Resinp = 1298
#         elif self.instrument == 'HST+Gemini':
#             Resinp = 129+305

#         if embedding:
#             self.embedding = nn.Sequential(
#                 SoftClip(100.0),
#                 ResMLP(
#                     Resinp , self.emb_output , #1298 + 129 / 305
#                     hidden_features= d.tolist(), #[512] * 3 + [256] * 5 + [128] * 7, #
#                     activation=nn.ELU,
#                 ),
#             )
#         else:
#             self.embedding = nn.Sequential(
#                 SoftClip(100.0),
#             )
        
#         l, u = torch.tensor(self.LOWER), torch.tensor(self.UPPER)

#         self.npe = NPE(
#             self.no_of_params, self.emb_output,
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms= self.transforms,
#             build=NAF,
#             signal = self.signal,
#             hidden_features= self.hidden_features * self.no_of_hidden_features,
#             activation='ELU',
#         )

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         return self.npe(theta, self.embedding(x))

#     def flow(self, x: Tensor):  # -> Distribution
# #         print(x.size(), 'inside flow')
#         #print(self.embedding(x).size(), )
#         return self.npe.flow(self.embedding(x))

# class BNPELoss(nn.Module):
#     def __init__(self, estimator, prior, lmbda=100.0):
#         super().__init__()
#         self.estimator = estimator
#         self.prior = prior
#         self.lmbda = lmbda
#     def forward(self, theta, x):
#         theta_prime = torch.roll(theta, 1, dims=0)
#         log_p, log_p_prime = self.estimator(
#             torch.stack((theta, theta_prime)),
#             x,
#         )
#         l0 = -log_p.mean()
#         lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
#         return l0 + self.lmbda * lb


# class NPEWithoutEmbedding_sepEmb(nn.Module):
#     def __init__(self, 
#                  hf_miri= [2,3,5], 
#                  hf_inst= [3,5,7], 
#                  instrument= 'HST', 
#                  hidden_features = [512],
#                  no_of_hidden_features= 5,
#                  emb_miri_output = 64 ,
#                  emb_inst_output = 8 ,
#                  no_of_params = 29,
#                  transforms = 5, 
#                 signal = 16,
#                 LOWER = None, 
#                 UPPER = None, 
#                 ):
        
#         super().__init__()
        
#         self.hf_miri = hf_miri
#         self.hf_inst = hf_inst
#         self.instrument = instrument
#         self.hidden_features = hidden_features
#         self.no_of_hidden_features= no_of_hidden_features
#         self.emb_miri_output = emb_miri_output
#         self.emb_inst_output = emb_inst_output
#         self.no_of_params = no_of_params
#         self.transforms = transforms
#         self.signal = signal
#         self.LOWER = LOWER
#         self.UPPER = UPPER
            
#         ## building the hidden_features array
#         x = torch.Tensor([[512] , [256] , [128]]).int()
#         c = torch.Tensor([]).int()
#         d = torch.Tensor([]).int()
#         for i in range(3):
#             c = torch.hstack( (c, x[i].repeat(self.hf_miri[i])) )
#             d = torch.hstack( (d, x[i].repeat(self.hf_inst[i])) )
                    
#         if self.instrument == 'HST':
#             Resinp = 129
#         elif self.instrument == 'Gemini':
#             Resinp = 305
#         elif self.instrument == 'HST+Gemini':
#             Resinp = 434
            
#         self.embedding_miri = nn.Sequential(
#             SoftClip(100.0),
#             ResMLP(
#                 1298 , self.emb_miri_output , #1298 + 129 / 305
#                 hidden_features= c.tolist(), #[512] * 3 + [256] * 5 + [128] * 7, #
#                 activation=nn.ELU,
#             ),
#         )

        
#         self.embedding_inst = nn.Sequential(
#         SoftClip(100.0),
#         ResMLP(
#             Resinp , self.emb_inst_output, #1298 + 129 / 305
#             hidden_features=  d.tolist(), #[512] * 3 + [256] * 5 + [128] * 7, #
#             activation=nn.ELU,
#         ),
#     )
        
#         l, u = torch.tensor(self.LOWER), torch.tensor(self.UPPER)

#         self.npe = NPE(
#             self.no_of_params, self.emb_inst_output + self.emb_miri_output ,
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms= self.transforms,
#             build=NAF,
#             signal = self.signal,
#             hidden_features= self.hidden_features * self.no_of_hidden_features,
#             activation='ELU',
#         )

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         return self.npe(theta, x )
        

#     def flow(self, x: Tensor):  # -> Distribution
#         return self.npe.flow(x)
    
# class NPEWithEmbedding_sepEmb_diffdim(nn.Module):
#     def __init__(self, 
#                  channels=['inst', 'chA', 'miri', 'chB'],
#                  hf_dict={'miri': [2, 3, 5], 'inst': [3, 5, 7], 'chA': [3, 4, 6], 'chB': [2, 3, 5]},
#                  dim_dict={'miri': 1298, 'inst': 434, 'chA': 300, 'chB': 200},
#                  emb_output_dict={'miri': 64, 'inst': 8, 'chA': 16, 'chB': 16},
#                  instrument='HST', 
#                  hidden_features=[512],
#                  no_of_hidden_features=5,
#                  no_of_params=29,
#                  transforms=5, 
#                  signal=16,
#                  LOWER=None, 
#                  UPPER=None):
        
#         super().__init__()
        
#         self.channels = channels
#         self.hf_dict = hf_dict
#         self.dim_dict = dim_dict
#         self.emb_output_dict = emb_output_dict
#         self.instrument = instrument
#         self.hidden_features = hidden_features
#         self.no_of_hidden_features = no_of_hidden_features
#         self.no_of_params = no_of_params
#         self.transforms = transforms
#         self.signal = signal
#         self.LOWER = LOWER
#         self.UPPER = UPPER
        
#         x = torch.Tensor([[512], [256], [128]]).int()
#         self.embeddings = nn.ModuleDict()
        
#         for ch in self.channels:
#             hidden_layers = torch.hstack([x[i].repeat(self.hf_dict[ch][i]) for i in range(3)])
#             self.embeddings[ch] = nn.Sequential(
#                 SoftClip(100.0),
#                 ResMLP(
#                     self.dim_dict[ch], self.emb_output_dict[ch],
#                     hidden_features=hidden_layers.tolist(),
#                     activation=nn.ELU,
#                 ),
#             )
        
#         l, u = torch.tensor(self.LOWER), torch.tensor(self.UPPER)
#         self.npe = NPE(
#             self.no_of_params, 
#             sum(self.emb_output_dict[ch] for ch in self.channels),
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms=self.transforms,
#             build=NAF,
#             signal=self.signal,
#             hidden_features=self.hidden_features * self.no_of_hidden_features,
#             activation='ELU',
#         )
    
#     def extract_channel_slices(self, x):
#         start = x.shape[1]  # Total feature count
#         channel_slices = []
        
#         for ch in reversed(self.channels):  # Reverse to extract in order
#             end = start
#             start = end - self.dim_dict[ch]  # Compute new start
#             channel_slices.append(self.embeddings[ch](x[:, start:end]))  # Apply embedding
        
#         return torch.hstack(tuple(reversed(channel_slices)))  # Reverse back for correct order

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         stacked_embeddings = self.extract_channel_slices(x)
#         return self.npe(theta, stacked_embeddings)
        
#     def flow(self, x: Tensor):
#         stacked_embeddings = self.extract_channel_slices(x)
#         return self.npe.flow(stacked_embeddings)

  
# class NPEWithEmbedding_oneEmbstd(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.embedding = nn.Sequential(
#             ResMLP(
#                 1603 , 64, #1298 + 129
#                 hidden_features=[512] * 2 + [256] * 3 + [128] * 5,
#                 activation=nn.ELU,
#             ),
#         )
        
#         l, u = torch.tensor(LOWER), torch.tensor(UPPER)

#         self.npe = NPE(
#             31, 64,
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms=3,
#             build=NAF,
#             hidden_features=[512] * 5,
#             activation='ELU',
#         )

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         return self.npe(theta, self.embedding(x))

#     def flow(self, x: Tensor):  # -> Distribution
#         return self.npe.flow(self.embedding(x))

# class NPEWithEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.embedding = nn.Sequential(
#             SoftClip(100.0),
#             ResMLP(
#                 1296, 64,
#                 hidden_features=[512] * 2 + [256] * 3 + [128] * 5,
#                 activation=nn.ELU,
#             ),
#         )

#         l, u = torch.tensor(LOWER), torch.tensor(UPPER)

#         self.npe = NPE(
#             23, 64,
#             moments=((l + u) / 2, (u - l) / 2),
#             transforms=3,
#             build=NAF,
#             hidden_features=[512] * 5,
#             activation='ELU',
#         )

#     def forward(self, theta: Tensor, x: Tensor) -> Tensor:
#         return self.npe(theta, self.embedding(x))

#     def flow(self, x: Tensor):  # -> Distribution
# #         print(x.size(), 'inside flow')
#         #print(self.embedding(x).size(), )
#         return self.npe.flow(self.embedding(x))