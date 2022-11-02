import os
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from sklearn.model_selection import KFold

import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Dataset(torch.utils.data.Dataset):
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

'''
class DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data

    def special_tokenizing(self, dataframe) :
        data = []

        # add special_tokens
        special_tokens_dict = {'additional_special_tokens': ['[rtt]','[sampled]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )

            if 'rtt' in item['source'] :
                text = "[rtt]" + text
            else :
                text = "[sampled]" + text

            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=4
        )
'''

#class KfoldDataLoader(pl.LightningDataModule):
class DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        bce,
        k,
        split_seed,  
        num_splits
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.bce = bce
        self.k = k #fold number
        self.split_seed = split_seed # split needs to be always the same for correct cross validation
        self.num_splits = num_splits

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = "[SEP]".join( # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
                [item[text_column] for text_column in self.text_columns]
            )
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data
        
    def special_tokenizing(self, dataframe) :
        data = []

        # add special_tokens
        special_tokens_dict = {'additional_special_tokens': ['[rtt]','[sampled]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )

            if 'rtt' in item['source'] :
                text = "[rtt]" + text
            else :
                text = "[sampled]" + text

            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data
    
    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 데이터 준비
            total_data = pd.read_csv(self.train_path)
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터 셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            # train + dev

            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else: # 평가 데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=4
        )
