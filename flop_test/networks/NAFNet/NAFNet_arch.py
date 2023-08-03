# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append('C:/Users/USER/Desktop/Codes/VDNet/')
import os
print(os.getcwd())
print(sys.path)
# from .arch_util import LayerNorm2d
from timm.models.layers import create_attn, get_attn, create_classifier



import math
import torch
from torch import nn as nn
from torch.nn import functional as F
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        # use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes) if norm_layer is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width) if norm_layer is not None else nn.Identity()
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        # self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes) if norm_layer is not None else nn.Identity()

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        # x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x

class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x.contiguous(), 'b c d h w -> b d h w c')
        x = super().forward(x)
        return rearrange(x.contiguous(), 'b d h w c -> b c d h w')

class ConvNeXT3D(nn.Module):
    def __init__(self, in_channels):
        super(ConvNeXT3D, self).__init__()
        self.conv3d_7x7 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,7,7), padding=(1,3,3), stride=(1,1,1), groups=in_channels)
        self.norm  = LayerNorm3d(in_channels)
        self.conv3d_1x1_1 = nn.Conv3d(in_channels, in_channels*4, 1, padding=0, stride=1, groups=1)
        self.act = nn.GELU()
        self.conv3d_1x1_2 = nn.Conv3d(in_channels *4, in_channels, 1, padding=0, stride=1, groups=1)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        res = x
        x = self.conv3d_7x7(x)
        x = self.norm(x)
        x = self.conv3d_1x1_1(x)
        x = self.act(x)
        x = self.conv3d_1x1_2(x)
        x = x + res
        return x.permute(0,2,1,3,4).contiguous()
    

class InceptionDWConv3d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv3d(gc, gc, (5,3,3), padding=(5//2,square_kernel_size//2,square_kernel_size//2), groups=gc)
        self.dwconv_w = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0,0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0,band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )
    
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)

        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=False)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x
    

class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv3d,
            norm_layer=nn.BatchNorm3d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x.permute(0,2,1,3,4).contiguous()


        

class VNAFNeXt(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.en_convnext3ds = nn.ModuleList()
        self.mi_convnext3ds = nn.ModuleList()
        self.de_convnext3ds = nn.ModuleList()

        

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                    
                )
            )
            self.en_convnext3ds.append(
                nn.Sequential(
                    *[MetaNeXtBlock(chan) for _ in range(num)]   
                )   
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        self.mi_convnext3ds = \
            nn.Sequential(
                *[MetaNeXtBlock(chan) for _ in range(middle_blk_num)]
            )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.de_convnext3ds.append(
                nn.Sequential(
                    *[MetaNeXtBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
    def forward(self, videos):
        
        _,_,_,H_original, W_original = videos.shape

        videos = self.check_video_size(videos)
        B, T, C, H, W = videos.shape
        inp = videos.contiguous().view(B*T, C, H, W)

        x = self.intro(inp) # B*T, chan, H, W

        

        # x = x.contiguous().view(B, T, C, H, W)

        encs = []

        for encoder, en_convnext3d, down in zip(self.encoders, self.en_convnext3ds, self.downs):
            C,H,W = x.shape[1:]
            x = encoder(x) # B * T, C, H, W
            x = x.contiguous().view(B, T, C, H, W)
            x = en_convnext3d(x) # B, T, C, H, W
            x = x.contiguous().view(B * T, C, H, W)
            encs.append(x)
            x = down(x)
        
        C,H,W = x.shape[1:]
        x = self.middle_blks(x)
        x = x.contiguous().view(B, T, C, H, W)
        x = self.mi_convnext3ds(x) # B, T, C, H, W

        x = x.contiguous().view(B * T, C, H, W)


        for decoder, de_convnext3d, up, enc_skip in zip(self.decoders, self.de_convnext3ds, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            C,H,W = x.shape[1:]
            x = x.contiguous().view(B, T, C, H, W)
            x = de_convnext3d(x) # B, T, C, H, W
            x = x.contiguous().view(B * T, C, H, W)

        x = self.ending(x)
        # x = x + inp
        C,H,W = x.shape[1:]
        x = x.contiguous().view(B, T, C , H, W).float()

        return x[:,:,:, :H_original, :W_original]
    def check_video_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# class VNAFNeXtLocal(Local_Base, VNAFNeXt):
#     def __init__(self, *args, train_size=(1, 5, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         VNAFNeXt.__init__(self, *args, **kwargs)
#
#         N, _, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

# class VNAFNetLocal(Local_Base, VNAFNet):
#     def __init__(self, *args, train_size=(1, 5, 3, 384, 384), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         VNAFNet.__init__(self, *args, **kwargs)
#
#         N, _, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
#
# class NAFNetLocal(Local_Base, NAFNet):
#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         NAFNet.__init__(self, *args, **kwargs)
#
#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    # import os
    #
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    img_channel = 3
    width = 32


    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 6
    dec_blks = [2, 2, 2, 2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = VNAFNeXt(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)#.cuda()
    # if torch.__version__[0] == '2':
    #     net = torch.compile(net)

    inp_shape = (5, 3, 256, 256)
    # print(net)
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

    flops, params = get_model_complexity_info(
        net,
        (5, 3, 512, 512),
        as_strings=False,
        print_per_layer_stat=False,
    )
    flops, params = flops_to_string(flops), params_to_string(params)
    print(": ", "FLOPs: ", flops, "#Params: ", params)


    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    print('macs:{}, params:{}'.format(macs, params))
    params = float(params[:-3])
    macs = float(macs[:-4])


    print('macs:{}, params:{}'.format(macs, params))

    inputs = torch.randn([2, 5, 3, 128, 128])
    out = net(inputs)
    print(out.shape)

    

