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
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load("model.pt")

    print("⚡ get trainer")
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=config.max_epoch, log_every_n_steps=1
    )
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("./data/submission_format.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)


if __name__ == "__main__":
    seed_everything()
    config = config

    main(config)
