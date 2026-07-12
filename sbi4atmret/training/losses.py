
import torch.nn as nn
from lampe.inference import NPELoss as LampeNPELoss
import torch


class BNPELoss(nn.Module):
    def __init__(self, estimator, prior, lmbda=100.0):
        '''
        Args:
            estimator: The full estimator model (flow + embedding).
                       .
            prior: The prior distribution.
            lmbda: The regularization parameter.
        '''
        super().__init__()
        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda

    def forward(self, theta, x):
        """
        Compute the BNPE loss.

        Args:
            theta: Parameters (target variables)
            x: Observations as a dictionary of tensors

        Returns:
            Loss value (combination of negative log likelihood and regularization term)
        """
        theta_prime = torch.roll(theta, 1, dims=0)
        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )
        l0 = -log_p.mean()
        lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
        return l0 + self.lmbda * lb
        

class NPELoss(nn.Module):
    def __init__(self, estimator, prior):
        """
        Wrapper around Lampe's NPELoss.

        Args:
            estimator: The full estimator model (flow + embedding).
                       Must implement forward(theta, x) as expected by Lampe.
        """
        super().__init__()

        # Store the estimator (not strictly needed, but useful for debugging / extensions)
        self.estimator = estimator

        # Instantiate Lampe's loss ONCE.
        # Avoid creating this inside forward(), which would be inefficient.
        self.loss_fn = LampeNPELoss(self.estimator)

    def forward(self, theta, x):
        """
        Compute the NPE loss.

        Args:
            theta: Parameters (target variables)
            x: Observations as a dictionary of tensors

        Returns:
            Loss value (typically negative log likelihood)
        """

        # Delegate to Lampe's implementation
        return self.loss_fn(theta, x)