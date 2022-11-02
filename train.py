from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch
from dataloader.dataloader import DataLoader
from model.model import Model
from trainer.trainer import Trainer
import pytorch_lightning as pl
import functools
import random
import os
import numpy as np


def rsetattr(obj, attr, val):
    """
    recursion을 이용하여,
    .형태로 구조화되어있는 값 수정하기."""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


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


def main(cfg):
    seed_everything(cfg.train.seed)

    exp_name = "_".join(
        map(
            str,
            [
                cfg.name,
                cfg.model.model_name,
                cfg.optimizer.name,
                cfg.lr_scheduler.name,
                cfg.train.learning_rate,
                cfg.train.batch_size,
            ],
        )
    )
    wandb_logger = WandbLogger(name=exp_name, project=cfg.project)

    dataloader = DataLoader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.dev_path,
        cfg.path.test_path,
        cfg.path.predict_path,
    )
    model = Model(cfg)

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.train.gpus,
        max_epochs=cfg.train.max_epoch,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.logging_step,
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f"{cfg.model.saved_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)

    args, options = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    for option in options:
        arg_name, value = option.split("=")
        try:  # value가 int인지, float인지, string인지 체크
            value = int(value) if float(value) == int(float(value)) else float(value)
        except:
            pass
        # options에 추가로 적용한 args를 적용.
        rsetattr(cfg, arg_name, value)
    main(cfg)
