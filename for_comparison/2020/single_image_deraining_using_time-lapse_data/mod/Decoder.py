import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import deconv_block


class Decoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.deconv3 = deconv_block(
            kernel_size=3, in_channels=256, out_channels=256)
