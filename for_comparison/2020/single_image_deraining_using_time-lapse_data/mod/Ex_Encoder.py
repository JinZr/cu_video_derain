import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import conv_block


class ExEncoder(nn.Module):

    def __init__(self, kernel_size: int) -> None:
        super().__init__()

        self.conv1_1 = conv_block(kernel_size, 1, 64)
        self.conv1_2 = conv_block(kernel_size, 64, 64)
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )
        self.conv2_1 = conv_block(kernel_size, 64, 128)
        self.conv2_2 = conv_block(kernel_size, 128, 128)
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )
        self.conv3_1 = conv_block(kernel_size, 128, 256)
        self.conv3_2 = conv_block(kernel_size, 256, 256)
        self.conv3_3 = conv_block(kernel_size, 256, 256)
        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        self.block = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.max_pool_1,
            self.conv2_1,
            self.conv2_2,
            self.max_pool_2,
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.max_pool_3
        ])

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    ExEncoder()
