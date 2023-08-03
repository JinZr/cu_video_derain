import argparse

import pyiqa
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid_metric = pyiqa.create_metric("fid")
score = fid_metric("")
print("fid", score)


niqe_metric = pyiqa.create_metric("niqe")
score = niqe_metric("")
print("niqe", score)
