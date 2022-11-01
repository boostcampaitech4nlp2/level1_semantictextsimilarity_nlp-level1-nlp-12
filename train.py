import os
import random
import numpy as np

import torch

import argparse
from models.model import Model
from dataloader.dataloader import DataLoader
from trainer.trainer import Trainer

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# set seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def main(config):
    exp_name = "_".join(
        map(
            str,
            [
                config.wandb.name,
                config.wandb.info,
                config.model.name,
                config.optimizer,
                config.scheduler,
                config.train.learning_rate,
                config.train.batch_size,
            ],
        )
    )
    wandb_logger = WandbLogger(name=exp_name, project=config.wandb.project)

    print("⚡ get dataloader")
    dataloader = DataLoader(
        config.model.name,
        config.train.batch_size,
        config.data.shuffle,
        config.path.train_path,
        config.path.dev_path,
        config.path.test_path,
        config.path.predict_path,
    )

    print("⚡ get model")
    model = Model(config)

    trainer = Trainer(config, wandb_logger)

    print("⚡ Training Start ⚡")
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # torch.save(model, "model.pt")
    torch.save(model, f"{config.model.saved_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")

    # sweep- 하이퍼파라미터 튜닝 인자
    parser.add_argument("--sweep", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)

    args, _ = parser.parse_known_args()

    config = OmegaConf.load(f"./configs/{args.config}.yaml")
    print("⚡ config file: ", args.config)

    if args.sweep == 1:
        config.train.batch_size = args.batch_size

    seed_everything(config.train.seed)

    main(config)
