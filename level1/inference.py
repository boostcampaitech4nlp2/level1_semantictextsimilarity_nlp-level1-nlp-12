import os
import random
import pandas as pd
import numpy as np

import torch

from configs.configs import config
from dataloader.dataloader import DataLoader

import pytorch_lightning as pl


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
    pl.seed_everything(seed, workers=True)


def main(config):
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
    model = torch.load("model.pt")

    print("⚡ get trainer")
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=config.max_epoch, log_every_n_steps=1
    )
    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output = pd.read_csv("./data/submission_format.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)


if __name__ == "__main__":
    seed_everything()
    config = config

    main(config)
