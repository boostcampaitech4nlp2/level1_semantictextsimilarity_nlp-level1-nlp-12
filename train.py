import os
import random
import numpy as np

import torch

from configs.configs import config
from models.model import Model
from dataloader.dataloader import DataLoader
from trainer.trainer import Trainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# set seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    exp_name = "_".join(
        [
            config.name,
            config.info,
            config.optimizer,
            config.scheduler,
            str(config.learning_rate),
            str(config.batch_size),
        ]
    )
    wandb_logger = WandbLogger(name=exp_name, project=config.wandb_project)

    print("⚡ get dataloader")
    dataloader = DataLoader(
        config.model_name,
        config.batch_size,
        config.shuffle,
        config.train_path,
        config.dev_path,
        config.test_path,
        config.predict_path,
    )

    print("⚡ get model")
    model = Model(config)

    print("⚡ get trainer")
    trainer = Trainer(config, wandb_logger).trainer

    print("⚡ Training Start ⚡")
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, "model.pt")


if __name__ == "__main__":
    seed_everything()
    config = config

    main(config)
