import argparse
import collections
import torch
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
import pytorch_lightning as pl
import os
import pandas as pd

## https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

# fix random seeds for reproducibility
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저 경고떠서 추가한 부분
pl.seed_everything(SEED, workers=True)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    # setup data_loader instances
    data_loader = config.init_obj(
        "data_loader", module_data
    )  # contains train/dev/test dataloaders

    # get function handles of loss and metrics
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=1, log_every_n_steps=1
    )

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(config["model_save_path"])
    predictions = trainer.predict(model=model, datamodule=data_loader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("/opt/ml/data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("/opt/ml/data/output.csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
