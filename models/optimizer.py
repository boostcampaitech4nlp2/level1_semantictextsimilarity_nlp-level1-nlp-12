import torch


def get_optimizer(parameters, config):
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=config.train.learning_rate)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=config.train.learning_rate)
    return optimizer


def get_scheduler(optimizer, config):

    if config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return scheduler
