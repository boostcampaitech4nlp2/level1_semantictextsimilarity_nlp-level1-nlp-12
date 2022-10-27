from base import BaseDataLoader
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class STS_Dataloader(BaseDataLoader):
    def __init__(
        self,
        checkpoint,
        batch_size,
        shuffle,
        num_workers,
        train_path,
        val_path,
        test_path,
        predict_path,
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            predict_path=predict_path,
            **kwargs
        )
        self.checkpoint = checkpoint
        #self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def setup(self, stage='fit'):
        if stage == 'fit':
            # read and assign csv files
            train_data = super().get_data('train')
            val_data = super().get_data('dev')

            # preprocess train/val data
            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            super().load_dataset('train', CustomDataset(train_inputs, train_targets))
            super().load_dataset('val', CustomDataset(val_inputs, val_targets))

        else:
            # 평가데이터 준비
            test_data = super().get_data('test')
            predict_data = super().get_data('predict')

            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)

            super().load_dataset('test', CustomDataset(test_inputs, test_targets))
            super().load_dataset('predict', CustomDataset(predict_inputs, predict_targets))
