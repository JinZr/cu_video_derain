import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Ex_Encoder
import FilterGeneratingNetwork


class DerainModule(nn.Module):

    def __init__(self) -> None:
        super(DerainModule, self).__init__()

        self.e3 = Ex_Encoder.ExEncoder(kernel_size=3, padding=1)
        self.e5 = Ex_Encoder.ExEncoder(kernel_size=5, padding=2)
        self.e7 = Ex_Encoder.ExEncoder(kernel_size=7, padding=3)

        self.filter_generating_network = FilterGeneratingNetwork.FilterGeneratingNetwork(
            in_channels=768,
            mk_filter_dim=256,
            out_channels=768
        )

    def forward(self, x):
        e3_feat_b1, e3_feat_b2, e3_feat_b3 = self.e3(x)
        _, _, e5_feat_b3 = self.e5(x)
        _, _, e7_feat_b3 = self.e7(x)

        e3_e5_e7 = torch.concat([
            e3_feat_b3,
            e5_feat_b3,
            e7_feat_b3
        ])

        generated_filter = self.filter_generating_network(e3_e5_e7)

        filtered_e3_e5_e7 = generated_filter * e3_e5_e7


if __name__ == '__main__':
    import numpy as np
    module = DerainModule().float()
    module(
        torch.from_numpy(np.random.rand(32, 3, 128, 128)).float()
    )
