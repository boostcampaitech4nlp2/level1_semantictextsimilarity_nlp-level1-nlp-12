import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)
