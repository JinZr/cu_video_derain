import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.parallel
from PIL import Image
from torchvision import transforms

from .models import modules
from .models import net
from .models.backbone_dict import backbone_dict
from .models.net import FineNet, FineNet_npic


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==== model utils ====
def build_model(args):
    if args.backbone.startswith("resnet"):
        backbone = backbone_dict[args.backbone](
            use_bn=args.use_bn, model_dir=args.torch_home
        )
        encoder = modules.E_resnet(backbone, use_bn=args.use_bn)

        if args.backbone in ["resnet50"]:
            decoder = modules.D_resnet(num_features=2048)
            net_coarse = net.CoarseNet(
                encoder,
                decoder,
                block_channel=[256, 512, 1024, 2048],
                refinenet=args.refinenet,
                bidirectional=args.use_bilstm,
                input_residue=args.input_residue,
            )
        elif args.backbone in ["resnet18", "resnet34"]:
            decoder = modules.D_resnet(num_features=512)
            net_coarse = net.CoarseNet(
                encoder,
                decoder,
                block_channel=[64, 128, 256, 512],
                refinenet=args.refinenet,
                bidirectional=args.use_bilstm,
                input_residue=args.input_residue,
            )

    elif args.backbone.startswith("densenet"):
        backbone = backbone_dict[args.backbone]()
        encoder = modules.E_densenet(backbone)

        if args.backbone in ["densenet121"]:
            decoder = modules.D_densenet(num_features=1024, use_bn=True)
            net_coarse = net.CoarseNet(
                encoder,
                decoder,
                block_channel=[128, 256, 512, 1024],
                refinenet=args.refinenet,
                bidirectional=args.use_bilstm,
                input_residue=args.input_residue,
            )
        elif args.backbone in ["densenet169"]:
            decoder = modules.D_densenet(num_features=1664, use_bn=True)
            net_coarse = net.CoarseNet(
                encoder,
                decoder,
                block_channel=[128, 256, 640, 1664],
                refinenet=args.refinenet,
                bidirectional=args.use_bilstm,
                input_residue=args.input_residue,
            )

    else:
        raise ValueError("Unrecognized backbone model name {}.".format(args.backbone))

    # discriminator = net.C_C3D_1()

    if args.F_npic:
        net_fine = FineNet_npic()
    else:
        net_fine = FineNet()

    return net_coarse, net_fine


# ==== data utilities =====
__imagenet_stats = {
    "mean": torch.tensor([0.485, 0.456, 0.406]),
    "std": torch.tensor([0.229, 0.224, 0.225]),
}


# ==== metrics ====
def calculate_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = tensor2array(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2array(img2)

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def tensor2img(t):
    unnormalize = transforms.Normalize(
        (-__imagenet_stats["mean"] / __imagenet_stats["std"]).tolist(),
        (1.0 / __imagenet_stats["std"]).tolist(),
    )

    img = transforms.ToPILImage()(unnormalize(t).cpu())
    return img


def tensor2array(t):
    transform_unnorm = transforms.Normalize(
        (-__imagenet_stats["mean"] / __imagenet_stats["std"]).tolist(),
        (1.0 / __imagenet_stats["std"]).tolist(),
    )

    t = np.array(transforms.ToPILImage()(transform_unnorm(t).detach().cpu()))
    return t


# ==== misc ====
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_image(t, filename):
    img = tensor2img(t)
    img.save(filename)


def calculate_psnr_on_tensors(t1, t2):
    a1 = tensor2array(t1)
    a2 = tensor2array(t2)

    return calculate_psnr(a1, a2)


# clamp
min_rgb = torch.tensor([-2.1179, -2.0357, -1.8044]).unsqueeze(-1).unsqueeze(-1).cuda()
max_rgb = torch.tensor([2.2489, 2.4286, 2.6400]).unsqueeze(-1).unsqueeze(-1).cuda()


def clamp_on_imagenet_stats(t):
    t = t.transpose(1, 2)

    t = torch.where(t < min_rgb, min_rgb, t)
    t = torch.where(t > max_rgb, max_rgb, t)

    t = t.transpose(1, 2)

    return t


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
