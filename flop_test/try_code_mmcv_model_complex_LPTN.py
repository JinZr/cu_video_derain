# Benchmarks used to test the inference speed, number of parameters and the FLOPs

import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
import time
import argparse

import networks.RDD_Net.rdd as rdd
from networks.ESTINet.utils import build_model
from networks.ESTINet.trainopt import _get_train_opt
from networks.S2VD.networks.derain_net import DerainNet

# from networks.S2VD.networks.generators import GeneratorState, GeneratorRain
from networks.IDT.utils.model_utils import get_arch

# from timm.models import create_model
# from  networks.LPTN_paper_arch import LPTNPaper
# from networks.test_models_ours.Network_V1_6_plain_MI_oriRDB2_wLeakeyReLU import UNet
# from networks.Two_stages_models.MSBDN_RDFF import Net
# from networks.transweather.transweather_model import Transweather


def speed_test(model, img_size=None, max_iters=100):
    try:
        input = torch.rand(img_size).cuda()
        print(f"Input Feature Map: {input.shape}")
        for i in range(30):  # warmup
            output = model(input)

        torch.cuda.synchronize()
        start = time.time()
        for i in range(max_iters):
            output = model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("--------------------------" * 3)
        print(
            f"FPS: {max_iters / (end - start)}",
            f"avg time: {(end - start) / max_iters}",
        )
        print("--------------------------" * 3)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    model_name_list = ["rdd", "estinet", "s2vd", "idt"]
    # model_name_list = ["estinet", "s2vd", "idt"]
    enable_speed_test = True
    model_name_list = ["rdd"]

    for model_name in model_name_list:
        print("********************************************************")
        if model_name == "rdd":
            model = rdd.Net(
                num_channels=3,
                base_filter=256,
                feat=64,
                num_stages=3,
                n_resblock=3,
                nFrames=7,
                scale_factor=4,
            ).cuda()

            flops, params = get_model_complexity_info(
                model,
                # (3, 128, 888),
                (3, 512, 512),
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops, params = flops_to_string(flops), params_to_string(params)
            print(model_name, ": ", "FLOPs: ", flops, "#Params: ", params)
            if enable_speed_test:
                speed_test(model, img_size=(1, 1, 3, 512, 512))
            print("********************************************************\n")
            exit(0)
        elif model_name == "estinet":
            args = _get_train_opt()
            net_C, net_F = build_model(args)
            net_C.cuda()
            net_F.cuda()
            flops, params = get_model_complexity_info(
                net_C,
                (3, 5, 640, 360),
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops, params = flops_to_string(flops), params_to_string(params)
            print("net_C", ": ", "FLOPs: ", flops, "#Params: ", params)
            if enable_speed_test:
                speed_test(net_C, img_size=(1, 3, 5, 640, 360))

            flops, params = get_model_complexity_info(
                net_F,
                (3, 5, 320, 180),
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops, params = flops_to_string(flops), params_to_string(params)
            print("net_F", ": ", "FLOPs: ", flops, "#Params: ", params)
            if enable_speed_test:
                speed_test(net_F, img_size=(1, 3, 5, 320, 180))

            print("********************************************************\n")
        elif model_name == "s2vd":
            latent_size, state_size, motion_size, feature_state = 64, 128, 64, 128
            patch_size, feature_rain_G, feature_derain_D, n_resblocks = 64, 64, 32, 8
            DNet = DerainNet(
                n_features=feature_derain_D, n_resblocks=n_resblocks
            ).cuda()
            flops, params = get_model_complexity_info(
                DNet,
                (3, 5, 320, 180),
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops, params = flops_to_string(flops), params_to_string(params)
            print("DNet", ": ", "FLOPs: ", flops, "#Params: ", params)
            if enable_speed_test:
                speed_test(DNet, img_size=(1, 3, 5, 320, 180))
            print("********************************************************\n")
        elif model_name == "idt":
            parser = argparse.ArgumentParser()
            parser.add_argument("--arch", type=str, default="IDT")
            parser.add_argument("--in_chans", type=int, default=3)
            parser.add_argument("--embed_dim", type=int, default=32)
            parser.add_argument("--win_size", type=int, default=8)
            parser.add_argument(
                "--depths", type=int, nargs="+", default=[3, 3, 2, 2, 1, 1, 2, 2, 3]
            )
            parser.add_argument(
                "--num_heads",
                type=int,
                nargs="+",
                default=[1, 2, 4, 8, 16, 16, 8, 4, 2],
            )
            parser.add_argument("--mlp_ratio", type=float, default=4.0)
            parser.add_argument("--qkv_bias", type=bool, default=True)
            parser.add_argument(
                "--downtype",
                type=str,
                default="Downsample",
                help="Downsample|Shufflesample",
            )
            parser.add_argument(
                "--uptype",
                type=str,
                default="Upsample",
                help="Upsample|Unshufflesample",
            )
            parser.add_argument("--batch_size", type=int, default=4)
            parser.add_argument(
                "--shuffle", action="store_true", help="shuffle for dataloader"
            )
            parser.add_argument(
                "--crop_size", type=int, default=128, help="crop size for network"
            )
            parser.add_argument("--channel", type=int, default=32)
            parser.add_argument("--embed", type=int, default=32)
            opt = parser.parse_args()
            IDTNet = get_arch(opt).cuda()
            flops, params = get_model_complexity_info(
                IDTNet,
                (3, 128, 128),
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops, params = flops_to_string(flops), params_to_string(params)
            print("IDTNet", ": ", "FLOPs: ", flops, "#Params: ", params)
            if enable_speed_test:
                speed_test(IDTNet, img_size=(1, 3, 128, 128))
            print("********************************************************\n")
        else:
            raise NotImplementedError


# srun -p VC -N 1 --gres=gpu:1 --ntasks=1 --cpus-per-task=10 --quotatype=spot python model_benchmark.py
