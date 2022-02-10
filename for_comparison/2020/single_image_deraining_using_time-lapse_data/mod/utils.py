from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv_block(kernel_size: int, in_channels: int, out_channels: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            # padding_mode=
            padding=padding
        ),
        nn.BatchNorm2d(
            num_features=out_channels
        ),
        nn.ReLU(inplace=True)
    )


def deconv_block(kernel_size: int, in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size)
    )


def init_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    kernel_value: List[int]
) -> nn.Conv2d:
    with torch.no_grad():
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size)
        )
        conv.weight.data = kernel_value
    return conv
