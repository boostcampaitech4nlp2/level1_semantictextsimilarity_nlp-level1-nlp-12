import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


def get_loss_func(config):
    if config.loss_func == "L1Loss":
        loss_func = torch.nn.L1Loss()

    return loss_func


class TripletLoss(Function):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss
