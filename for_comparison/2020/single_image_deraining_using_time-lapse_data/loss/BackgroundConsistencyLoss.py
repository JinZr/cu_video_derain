import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def background_consistency_loss(batch):
    b, c, w, h = batch.shape
    