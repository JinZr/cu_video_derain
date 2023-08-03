import argparse

import pyiqa
import torch

path = ""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid_metric = pyiqa.create_metric("fid")
score = fid_metric(path)
print("fid", score)


niqe_metric = pyiqa.create_metric("niqe")
score = niqe_metric(path)
print("niqe", score)
