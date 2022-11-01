import argparse
import os
import collections
import wandb
import torch
import pandas as pd
import data_loader.data_loaders as module_data
import model.model as module_model
from parse_config import ConfigParser
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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
    logger.info(model_module)

    # custom wandb logger & checkpoint_callback
    ########### more info ##############
    # checkpoint: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
    # wandb : https://docs.wandb.ai/guides/integrations/lightning
    callback_monitor = configs.pop("callback_monitor", "val_loss")
    earlystop_callback = EarlyStopping(monitor=callback_monitor, patience=5)
    checkpoint_callback = ModelCheckpoint(
                monitor=callback_monitor,
                mode='min',
                save_top_k=2
            )
    
    wandb_name = f"wjl-{configs['optimizer']}-{configs['batch_size']}-{configs['lr']}"
    wandb_project = "sts"
    wandb.init(name=wandb_name, project=wandb_project)
    wandb.config.update({
        "callback_monitor":callback_monitor
    })
    wandb_logger = WandbLogger(log_model='all')

    # build trainer, then print to console
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = Trainer(
        accelerator=configs.pop('accelerator', 'auto'),
        max_epochs=configs.pop('max_epochs', 10),
        log_every_n_steps=configs.pop('log_every_n_steps', 1),
        deterministic=True, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        fast_dev_run=configs.pop('debugging_run', False)
    )
    logger.info(trainer)

    #if configs.pop('resume', None):
    trainer.fit(model=model_module, datamodule=dataloader_module)
    logger.info(f'Best checkpoint saved at {trainer.checkpoint_callback.best_model_path}')
    trainer.test(model=model_module, datamodule=dataloader_module)

    """# Inference
    predictions = trainer.predict(model=model_module, datamodule=dataloader_module)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./data/output.csv', index=False)"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str)
    parser.add_argument('-r', '--resume', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--ckpt', '--checkpoint'], type=str, target='checkpoint'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='batch_size'),
        CustomArgs(['--me', '--max_epochs'], type=int, target='max_epochs'),
        CustomArgs(['--dr', '--debugging_run'], type=bool, target='debugging_run')
    ]
    config_parser = ConfigParser.from_args(parser, options)

    main(config_parser)
