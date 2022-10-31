from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch


class SentTransformerDataset(Dataset):
    def __init__(self, inputs):
        pass

class CustomDataset(Dataset):
    '''baseline code'''
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])
            
    def __len__(self):
        return len(self.inputs)

    
class STSDataLoader(BaseDataLoader):
    '''augmented baseline code'''
    def __init__(self, **configs):
        super().__init__(**configs)
        
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def _tokenize(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])

        return data

    def _preprocess(self, data):
        data = data.drop(columns=self.delete_columns)
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        inputs = self._tokenize(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # read csv files
            self.train_data = self._read_data('train')
            self.val_data = self._read_data('val') # None if val_path is not given ins config.json

            # read and assign csv files
            train_inputs, train_targets = self._preprocess(self.train_data)
            self.train_dataset = CustomDataset(train_inputs, train_targets)

            if self.val_data is not None:
                val_inputs, val_targets = self._preprocess(self.val_data)
                self.val_dataset = CustomDataset(val_inputs, val_targets)
            else:
                self.train_dataset, self.val_dataset = self._split_validation_set(self.train_dataset)

        else:
            # read csv files
            self.test_data = self._read_data('test')
            self.predict_data = self._read_data('predict')

            test_inputs, test_targets = self._preprocess(self.test_data)
            predict_inputs, predict_targets = self._preprocess(self.predict_data)

            self.test_dataset = CustomDataset(test_inputs, test_targets)
            self.predict_dataset = CustomDataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            num_workers=self.num_workers
        ) # collate_fn = self.collate_fn,

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        ) # collate_fn = self.collate_fn,

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        ) # collate_fn = self.collate_fn,
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        ) # collate_fn = self.collate_fn,