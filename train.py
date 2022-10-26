import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
#import trainer.trainer as module_trainer
from parse_config import ConfigParser
import pytorch_lightning as pl
#from trainer import Trainer
#from utils import prepare_device
## https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

# fix random seeds for reproducibility
SEED = 12
pl.seed_everything(SEED, workers=True)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config):

    # setup data_loader instances
    dataloader = config.init_obj('data_loader', module_data, checkpoint=config['model']['args']['checkpoint']) # contains train/dev/test dataloaders

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']) 
    metric = getattr(module_metric, config['metric']) 

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_ftn('optimizer', torch.optim) # partial object. Will get an model instance in model.py
    lr_scheduler = config.init_ftn('lr_scheduler', torch.optim.lr_scheduler)

    # build model architecture, then print to console
    model = config.init_obj('model', module_model, criterion=criterion, metric=metric, optimizer=optimizer, lr_scheduler=lr_scheduler)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    #trainer = config.init_obj('Trainer', module_trainer)
    trainer = pl.Trainer(config['trainer'])
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch-Lightning Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: "./config.json")')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
