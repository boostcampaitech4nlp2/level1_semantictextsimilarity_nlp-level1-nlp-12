import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_func(config):
    if config.loss_func == "L1Loss":
        loss_func = torch.nn.L1Loss()

    return loss_func
