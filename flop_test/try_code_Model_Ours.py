import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
import time
import argparse

import networks.RDD_Net.rdd as rdd
from networks.ESTINet.utils import build_model
from networks.ESTINet.trainopt import _get_train_opt
from networks.S2VD.networks.derain_net import DerainNet
from networks.NAFNet.NAFNet_arch import VNAFNeXt

# from networks.S2VD.networks.generators import GeneratorState, GeneratorRain
from networks.IDT.utils.model_utils import get_arch

def speed_test(model, img_size=None, max_iters=100):
    try:
        input = torch.rand(img_size).cuda()
        print(f"Input Feature Map: {input.shape}")
        for i in range(30):  # warmup
            with torch.cuda.amp.autocast():
                output = model(input)

        torch.cuda.synchronize()
        start = time.time()
        for i in range(max_iters):
            with torch.cuda.amp.autocast():
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

def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))

if __name__ == "__main__":
    if torch.__version__[0] == '2':
        import torch._dynamo.config
        torch._dynamo.config.verbose=True
        torch._dynamo.config.suppress_errors = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_channel = 3
    width = 32


    enc_blks = [2, 2, 4, 6]
    middle_blk_num = 6
    dec_blks = [2, 2, 2, 2]
    net = VNAFNeXt(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).to(device)#.cuda()


    if torch.__version__[0] == '2':
        net = torch.compile(net)
    print_param_number(net)
    net.eval()
    speed_test(net, img_size=(1, 5, 3, 128, 128) , max_iters=200)
