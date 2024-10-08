import torch
import torch.nn as nn


class TensorStandardScaler(nn.Module):
    """
    Applies standard Gaussian normalization to :math:`\mathcal{N}(0, 1)`.

    Args:
        dim: Dimension to calculate mean and std.
    """
    def __init__(self, dim: int = 0):
        super(TensorStandardScaler, self).__init__()
        self.dim = dim
        self.mean, self.std = None, None

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and std to be used for later scaling.

        Args:
            x: Data used to compute the mean and standard deviation

        Returns:
            var_mean (Tuple[torch.Tensor, torch.Tensor]): Tuple of mean and std.
        """
        var_mean = torch.var_mean(x, dim=self.dim, correction=0)
        self.std, self.mean = var_mean
        self.std = self.std.sqrt()
        return var_mean

    def forward(self, x: torch.Tensor, with_mean: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The source tensor.
            with_mean: Whether to center the data before scaling.
        """
        if with_mean:
            x -= self.mean
        x /= (self.std + 1e-7)
        return x
