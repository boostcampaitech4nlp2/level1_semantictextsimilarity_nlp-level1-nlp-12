import argparse
import os
import collections
import wandb
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_model
from parse_config import ConfigParser
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


# fix random seeds for reproducibility
SEED = 42
seed_everything(SEED, workers=True) # sets seed for pytorch, numpy and python.random

def main(config_parser):
    os.environ['TOKENIZERS_PARALLELISM'] = "True"

    logger = config_parser.get_logger('train')
    configs = config_parser.config

    # setup data_loader and model
    dataloader_module = getattr(module_data, config_parser['data_loader'])(**configs)
    model_module = getattr(module_model, config_parser['model'])(**configs)

    # custom wandb logger & checkpoint_callback
    # more info: https://docs.wandb.ai/guides/integrations/lightning
    earlystop_callback = EarlyStopping(monitor="val_loss")

    wandb.init()
    wandb_name = f"{configs['optimizer']}-{configs['batch_size']}-{configs['lr']}"
    wandb_project = "sts"
    wandb_logger = WandbLogger(wandb_name, wandb_project)

    # build trainer, then print to console
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = Trainer(
        accelerator=configs.pop('accelerator', 'auto'),
        max_epochs=configs.pop('max_epochs', 10),
        log_every_n_steps=configs.pop('log_every_n_steps', 10),
        deterministic=True, 
        logger=wandb_logger,
        callbacks=[earlystop_callback]
    )
    logger.info(trainer)

    #if configs.pop('resume', None):
    trainer.fit(model=model_module, datamodule=dataloader_module)
    trainer.test(model=model_module, datamodule=dataloader_module)

    # Inference
    predictions = trainer.predict(model=model_module, datamodule=dataloader_module)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./data/output.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    """parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')"""
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-r', '--resume'], type=float, target='lr'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='batch_size'),
        CustomArgs(['--me', '--max_epochs'], type=int, target='max_epochs')
    ]
    config_parser = ConfigParser.from_args(parser, options)

    main(config_parser)
