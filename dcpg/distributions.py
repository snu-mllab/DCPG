import torch
from torch import Tensor
import torch.nn as nn

from dcpg.utils import init


class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        """
        Compute log probability
        """
        log_probs = super().log_prob(actions.squeeze(-1))
        log_probs = log_probs.view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        return log_probs

    def mode(self) -> Tensor:
        """
        Return mode
        """
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """

    def __init__(self, feature_dim: int, num_acitons: int):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01,
        )

        self.linear = init_(nn.Linear(feature_dim, num_acitons))

    def forward(self, x: Tensor) -> FixedCategorical:
        """
        Forward
        """
        x = self.linear(x)
        return FixedCategorical(logits=x)
