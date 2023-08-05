import glob
import os

import pyiqa
import torch
import torchvision.transforms as transforms
from ignite.contrib.metrics import *
from ignite.contrib.metrics.regression import *
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics import FID
from ignite.utils import *
from PIL import Image

# create default evaluator for doctests


def eval_step(engine, batch):
    return batch


default_evaluator = Engine(eval_step)


path = "/Users/zengruijin/Downloads/Ours/mid_motion"
gt_path = "/Users/zengruijin/Downloads/test_motion_frames"
imgs = glob.glob(f"{path}/*/*.png", recursive=True, root_dir=path)
dirs = os.listdir(path)
transform = transforms.Compose([transforms.PILToTensor()])


if __name__ == "__main__":
    import sys

    key = sys.argv[1]
    paths = {
        "Ours": "/Users/zengruijin/Downloads/Ours/mid_motion",
        "ECCV22": "/Users/zengruijin/Downloads/ECCV22/motion_ours",
        "nafnet": "/Users/zengruijin/Downloads/nafnet/CUHK_test_motion/",
        "S2VD": "/Users/zengruijin/Downloads/S2VD/S2VD-ours_final_motion",
        "IDT": "/Users/zengruijin/Downloads/IDT/motion",
        "ESTI_orig": "/Users/zengruijin/Downloads/ESTINet_orig/motion",
        "ESTI": "/Users/zengruijin/Downloads/ESTINet_Retrained/min_motion_ours",
    }
    for key in paths.keys():
        path = paths[key]
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        niqe_metric = pyiqa.create_metric("niqe")
        # fid_metric = FID(device=device, num_features=2048)
        # fid_metric = pytorch_fid.FID(device=device, num_features=2048)
        scores = []
        for dir in dirs:
            score = 0
            imgs = glob.glob(f"{path}/{dir}/*.png", recursive=True, root_dir=path)
            for img in imgs:
                img_name = img.split("/")[-1]
                # gt_img = f"{gt_path}/{dir}/{img_name}"
                # input_img = (
                #     transform(Image.open(img).convert("YCbCr")).unsqueeze(0).to(device)
                # ) / 255.0

                # print(input_img.shape)
                # exit()
                score += niqe_metric(img)
                # scores += fid_metric(f"{path}/{dir}", dataset_split="trainval")
            scores.append(score / len(imgs))
        import numpy as np

        print(f"{key} niqe", np.mean(scores))
        print(f"{key} niqe", scores)
