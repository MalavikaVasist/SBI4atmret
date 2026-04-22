from zuko.distributions import BoxUniform as ZukoBoxUniform
import torch

class BoxUniform(ZukoBoxUniform):
    def __init__(self, lower, upper):
        lower = torch.tensor(lower)
        upper = torch.tensor(upper)
        super().__init__(lower, upper)