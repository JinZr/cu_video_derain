import torch.nn as nn
import torch.nn.functional as F


class FilterGeneratingNetwork(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.mk_dyn_filter = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1
            ),
            # nn.AdaptiveAvgPool2d(
            # output_size=2
            # )
            # nn.Flatten(
            #     start_dim=2
            # )
        )

    def forward(self, x):
        x_mk_filter = self.mk_dyn_filter(x)
        pre_filter = F.adaptive_avg_pool2d(x_mk_filter, self.filter_size)
        b, c, h, w = x_mk_filter.shape
        x_mk_filter = x_mk_filter.view(1, b * c, h, w)
        pre_filter = pre_filter.view(
            b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x_mk_filter, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x_mk_filter, weight=pre_filter, groups=b * c)
        output = output.view(b, c, h, w)
        return output
