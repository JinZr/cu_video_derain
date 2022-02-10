import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DirectionalGradientNet(nn.Module):

    def __init__(self) -> None:
        super(DirectionalGradientNet, self).__init__()

    def forward(self, s):
        tan_s = None
