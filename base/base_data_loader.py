import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule


class BaseDataLoader(LightningDataModule):
    """
    Base class for all data loaders.
    Lightning data-module drills: `prepare_data` >> `set_up` >> `train/val/test_dataloader`

    """
    def __init__(self, **configs):
        super().__init__()
        self.checkpoint = configs['checkpoint']
        self.batch_size = configs.pop('batch_size', 16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            max_length=configs['max_length']
        )

        # dataloader-specific config
        dataloader_configs =  configs.pop('data_loader_args')
        self.shuffle = dataloader_configs.pop("shuffle", True)
        self.num_workers = dataloader_configs.pop("num_workers", 1)
        self.validation_split = dataloader_configs.pop("validation_split", 0.1)
        #self.collate_fn = getattr(torch.utils.data, dataloader_configs.pop("collate_fn", "default_collate"))
        
        self.train_path = dataloader_configs.pop("train_path", "")
        self.val_path = dataloader_configs.pop("val_path", "")
        self.test_path = dataloader_configs.pop("test_path", "")
        self.predict_path = dataloader_configs.pop("predict_path", "")

    def _read_data(self, stage):
        assert stage in ['train', 'val', 'dev', 'test', 'predict']
        if stage == 'dev':
            stage = 'val'
        path_map = {
            'train': self.train_path, 
            'val': self.val_path, 
            'test': self.test_path, 
            'predict': self.predict_path
        }
        path = path_map[stage]
        if len(path) > 0:
            return pd.read_csv(path)
        else:
            return None            

    def _split_validation_set(self, dataset):
        val_len = int(self.validation_split * len(dataset))
        train_len = len(dataset) - val_len
        return random_split(dataset, [train_len, val_len]) 
        