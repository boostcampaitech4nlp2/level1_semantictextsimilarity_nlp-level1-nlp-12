import argparse
import collections
import torch
import wandb
import os
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
import trainer.trainer as module_trainer
from parse_config import ConfigParser
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

## https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

# fix random seeds for reproducibility
SEED = 42
seed_everything(SEED, workers=True) # sets seed for pytorch, numpy and python.random

# to avoid parallelism error messages
os.environ['TOKENIZERS_PARALLELISM'] = "True" 

def main(config):

    # huggingface pretrained model name or checkpoint dir
    checkpoint = config['checkpoint']
    
    # setup train/dev/test data_loader instances
    dataloader = config.init_obj('data_loader', module_data, checkpoint=checkpoint) 

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_ftn('optimizer', torch.optim) # partial object. Will get an model instance in model.py
    lr_scheduler = config.init_ftn('lr_scheduler', torch.optim.lr_scheduler)

    # build model architecture, then print to console
    model = config.init_obj(
        'model', module_model, 
        checkpoint=checkpoint, 
        criterion=criterion, 
        metrics=metrics, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler
    )

    # custom wandb logger & checkpoint_callback
    # more info: https://docs.wandb.ai/guides/integrations/lightning
    checkpoint_callback = ModelCheckpoint(dirpath="./saved/model/", monitor="val_loss")
    wandb.init(name=config['wandb']['name'], project=config['wandb']['project'])
    wandb_args = {
        'batch_size': config['data_loader']['args']['batch_size'],
        'max_epochs': config['trainer']['args']['max_epochs'],
        'lr': config['optimizer']['args']['lr']
    }
    wandb.config.update(wandb_args)
    wandb_logger = WandbLogger()

    # build pl.trainer 
    trainer = config.init_obj(
        'trainer',
        module_trainer,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )
    trainer.fit(model=model, datamodule=dataloader)
    trainer.validate(model=model, datamodule=dataloader)
    #trainer.test(model=model, datamodule=dataloader)    
    
    """# save trained model

    # Inference
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./data/output.csv', index=False)"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch-Lightning Template')
    parser.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: "./config.json")')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--me', '--max_epochs'], type=int, target='trainer;args;max_epochs')
    ]
    config = ConfigParser.from_args(parser, options)
    main(config)
