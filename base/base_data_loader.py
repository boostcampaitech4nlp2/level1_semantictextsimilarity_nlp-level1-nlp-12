import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning import LightningDataModule

class BaseDataLoader(LightningDataModule):
    """
    Base class for all data loaders
    """
    def __init__(
        self, 
        batch_size,
        shuffle,
        num_workers,
        train_path,
        val_path,
        test_path,
        predict_path,
        **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.collate_fn = kwargs.pop('collate_fn', None)
        if self.collate_fn is None:
            self.collate_fn = default_collate

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def get_data(self, data_type):
        assert data_type in ['train', 'val', 'dev', 'test', 'predict']
        path_map = {
            'train': self.train_path, 
            'dev': self.val_path, 
            'val': self.val_path, 
            'test': self.test_path, 
            'predict': self.predict_path
        }
        return pd.read_csv(path_map[data_type])

    def load_dataset(self, data_type, dataset):
        '''
        Update `self.{data_type}_dataset` with a customized Dataset.
        '''
        if data_type == 'train':
            self.train_dataset = dataset
        elif data_type == 'val' or data_type == 'dev':
            self.val_dataset = dataset
        elif data_type == 'test':
            self.test_dataset = dataset
        elif data_type == 'predict':
            self.predict_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            collate_fn = self.collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, 
            collate_fn = self.collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, 
            collate_fn = self.collate_fn,
            num_workers=self.num_workers
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size, 
            collate_fn = self.collate_fn,
            num_workers=self.num_workers
        )


    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)"""
