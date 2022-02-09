import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Ex_Encoder


class DerainingNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.e3 = Ex_Encoder.ExEncoder(kernel_size=3)
        self.e5 = Ex_Encoder.ExEncoder(kernel_size=5)
        self.e7 = Ex_Encoder.ExEncoder(kernel_size=7)
