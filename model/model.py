from typing import Callable
import torch
import transformers
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        loss_func: Callable,
        metric: Callable,
        _optimizer: Callable,
        _lr_scheduler: Callable,
        optimizer_info,  # 임의로 일단 어떤 옵티마이저와 lr_scheduler을 사용할 지를 tracking하기위해 추가.
        lr_scheduler_info,
    ):
        super().__init__()
        # config에 들어가는 내용 -> Model의 파라미터가 들어가게하는 함수.언더바를 붙인 파라미터는 관리안한다.
        self.save_hyperparameters()

        self.model_name = model_name
        self.metric = metric
        self.optimizer = _optimizer
        self.lr_scheduler = _lr_scheduler
        self.lr = lr
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = loss_func
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            self.metric(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            self.metric(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]
