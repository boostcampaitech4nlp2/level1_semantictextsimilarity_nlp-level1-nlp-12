from omegaconf import OmegaConf
import argparse
import torch
from dataloader.dataloader import DataLoader
from trainer.trainer import Trainer
import pytorch_lightning as pl
import functools
import random
import os
import numpy as np
import pandas as pd
from model.model import Model


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

    dataloader = DataLoader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.dev_path,
        cfg.path.test_path,
        cfg.path.predict_path,
        cfg.train.bce,
        cfg.train.k,
        cfg.train.split_seed,
        cfg.train.num_splits,
    )
    model1 = Model.load_from_checkpoint("./checkpoint/epoch=15-step=16112.ckpt")
    model2 = Model.load_from_checkpoint(".")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.train.gpus,
        logger=False,
        max_epochs=cfg.train.max_epoch,
        log_every_n_steps=cfg.train.logging_step,
    )

    predictions1 = trainer.predict(model=model1, datamodule=dataloader)
    predictions2 = trainer.perdict(model=model2, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv("./data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("./data/output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)

    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    main(cfg)
