import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import deconv_block, conv_block


class Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.deconv = deconv_block()
        self.block1 = nn.Sequential(
            conv_block(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            conv_block(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                padding=0
            ),
            conv_block(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                padding=0
            ),
        )
        self.block2 = nn.Sequential(
            conv_block(
                kernel_size=3,
                in_channels=256,
                out_channels=128,
                padding=1
            ),
            conv_block(
                kernel_size=3,
                in_channels=128,
                out_channels=64,
                padding=1
            )
        )
        self.block3 = nn.Sequential(
            conv_block(
                kernel_size=3,
                in_channels=128,
                out_channels=64,
                padding=1
            ),
            conv_block(
                kernel_size=3,
                in_channels=64,
                out_channels=1,
                padding=1
            )
        )

    def forward(self, x, b3, b2, b1):
        x_b3 = torch.concat([x, b3], dim=1)
        deconv_x_b3 = self.deconv(x_b3)
        conv3 = self.block1(deconv_x_b3)
        conv3_1_b2 = torch.concat([conv3, b2], dim=1)
        deconv_conv3_1_b2 = self.deconv(conv3_1_b2)
        conv2 = self.block2(deconv_conv3_1_b2)
        conv2_1_b1 = torch.concat([conv2, b1], dim=1)
        deconv_conv2_1_b1 = self.deconv(conv2_1_b1)
        conv1 = self.block3(deconv_conv2_1_b1)
        return conv1
