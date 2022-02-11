import torch.nn as nn


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


def deconv_block() -> nn.UpsamplingBilinear2d:
    # return nn.ConvTranspose2d(
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     kernel_size=(kernel_size, kernel_size),
    #     stride=stride,
    #     padding=1,
    #     output_padding=1
    # )
    return nn.UpsamplingBilinear2d(
        scale_factor=2
    )
