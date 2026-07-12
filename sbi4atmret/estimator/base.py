import torch
import torch.nn as nn

class EstimatorBase(nn.Module):
    def __init__(self, flow: nn.Module, embedding: nn.Module):
        super().__init__()
        self.flow = flow
        self.embedding = embedding

    def forward(self, theta, x):
        """
        Compute log probability of theta given x.
        
        Args:
            theta: (B, D) or (K, B, D) parameter tensor
            x: (B, D_x) observation tensor
            
        Returns:
            log_prob: same leading shape as theta
        """
        x_emb = self.embedding(x.float())
        return self.flow(theta.float(), x_emb)

    def flow_forward(self, x):
        """Return the flow distribution conditioned on x."""
        x_emb = self.embedding(x.float())
        return self.flow.flow(x_emb)
    


