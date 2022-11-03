import os
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer

from torch.utils.data import Dataset

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
        """
        Args:
            dataframe (DataFrame): csv를 읽어온 데이터프레임

        Returns:
            data (List): 'sentence_1 + [SEP] + sentence_2'를 tokenization한 값들을 담은 list를 반환
        """
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

    def preprocessing(self, data):
        """데이터프레임을 읽어서 tokenizing된 inputs list와 labels list를 반환
        Args:
            data (DataFrame): csv를 읽어온 데이터프레임

        Returns:
            inputs, targets
        """
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


class ElectraDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return (
                torch.tensor(self.inputs[idx]["input_ids"]),
                torch.tensor(self.inputs[idx]["attention_mask"]),
            )
        else:
            return (
                torch.tensor(self.inputs[idx]["input_ids"]),
                torch.tensor(self.inputs[idx]["attention_mask"]),
                torch.tensor(self.targets[idx]),
            )

    def __len__(self):
        return len(self.inputs)


class ElectraDataLoader(pl.LightningDataModule):
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
            temp = dict()
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            temp["input_ids"] = outputs["input_ids"]
            temp["attention_mask"] = outputs["attention_mask"]
            data.append(temp)
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

            self.train_dataset = ElectraDataset(train_inputs, train_targets)
            self.val_dataset = ElectraDataset(val_inputs, val_targets)

        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = ElectraDataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = ElectraDataset(predict_inputs, [])

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


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx]["main_input_ids"]),
            torch.tensor(self.inputs[idx]["main_attention_mask"]),
            torch.tensor(self.inputs[idx]["pos_input_ids"]),
            torch.tensor(self.inputs[idx]["pos_attention_mask"]),
            torch.tensor(self.inputs[idx]["neg_input_ids"]),
            torch.tensor(self.inputs[idx]["neg_attention_mask"]),
        )

    def __len__(self):
        return len(self.inputs)


class ContrastiveDataLoader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path

        self.train_dataset = None
        self.val_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=160)

    def tokenizing(self, dataframe):
        data = []

        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            temp = dict()
            main_sent = item["main_sentence"]
            pos_pair = item["pos_sentence"]
            neg_pair = item["neg_sentence"]

            main_tokenized = self.tokenizer(
                main_sent,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
            pos_tokenized = self.tokenizer(
                pos_pair, add_special_tokens=True, padding="max_length", truncation=True
            )
            neg_tokenized = self.tokenizer(
                neg_pair,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )

            temp["main_input_ids"] = main_tokenized["input_ids"]
            temp["main_attention_mask"] = main_tokenized["attention_mask"]

            temp["pos_input_ids"] = pos_tokenized["input_ids"]
            temp["pos_attention_mask"] = pos_tokenized["attention_mask"]

            temp["neg_input_ids"] = neg_tokenized["input_ids"]
            temp["neg_attention_mask"] = neg_tokenized["attention_mask"]

            data.append(temp)
        return data

    def preprocessing(self, dataframe):
        inputs = self.tokenizing(dataframe)

        return inputs

    def setup(self, stage="fit"):
        if stage == "fit":
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs = self.preprocessing(train_data)
            val_inputs = self.preprocessing(val_data)

            self.train_dataset = ContrastiveDataset(train_inputs)
            self.val_dataset = ContrastiveDataset(val_inputs)

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
