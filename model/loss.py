import torch.nn.functional as F
from sentence_transformers import losses

def nll_loss(output, target):
    return F.nll_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)