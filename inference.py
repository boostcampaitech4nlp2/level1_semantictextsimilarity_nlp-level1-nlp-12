import os
import random
import pandas as pd
import numpy as np

import torch

import argparse
from omegaconf import OmegaConf
from dataloader.dataloader import DataLoader
from models.model import Model

import pytorch_lightning as pl


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
    print("\033[32m" + "⚡ get dataloader" + "\033[0m")
    dataloader = DataLoader(
        config.model.name,
        config.train.batch_size,
        config.data.shuffle,
        config.path.train_path,
        config.path.dev_path,
        config.path.test_path,
        config.path.predict_path,
    )

    print("\033[32m" + "⚡ get model" + "\033[0m")
    # model = torch.load(f"{config.model.saved_name}.pt")
    model = Model.load_from_checkpoint(config.model.model_ckpt_path)

    print("\033[32m" + "⚡ get trainer" + "\033[0m")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epoch,
        log_every_n_steps=1,
    )
    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv("./data/submission_format.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    seed_everything(config.train.seed)

    main(config)
