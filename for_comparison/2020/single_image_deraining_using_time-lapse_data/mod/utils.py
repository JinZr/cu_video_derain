import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv_block(kernel_size: int, in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.ModuleList([
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size)
        ),
        nn.BatchNorm2d(
            num_features=out_channels
        ),
        nn.ReLU(inplace=True)
    ])
    