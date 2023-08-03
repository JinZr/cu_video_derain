import glob
import os

import pyiqa
import torch

path = "/Users/zengruijin/Downloads/Ours/mid_motion"
imgs = glob.glob(f"{path}/*/*.png", recursive=True, root_dir=path)
dirs = os.listdir(path)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fid_metric = pyiqa.create_metric("fid")
    scores = 0
    for dir in dirs:
        scores += fid_metric(f"{path}/{dir}", dataset_split="trainval")
    print("fid", scores)

    niqe_metric = pyiqa.create_metric("niqe")
    score = niqe_metric(path)
    print("niqe", score)
