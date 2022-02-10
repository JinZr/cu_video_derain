import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FilterGeneratingNetwork(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mk_filter_dim: int
    ) -> None:
        super().__init__()

        self.mk_dyn_filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mk_filter_dim,
            kernel_size=(1, 1),
            stride=1
        )

    def forward(self, x):
        return self.mk_dyn_filter(x)
