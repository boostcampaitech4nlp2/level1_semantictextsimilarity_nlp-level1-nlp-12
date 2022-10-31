import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from functools import partial


class Model(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.learning_rate = config.train.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        # 로스함수도 하드코딩 말고 config에 사용 가능하게 수정완료.
        self.loss_func = getattr(torch.nn, config.loss)
        self.optimizer_info = config.optimizer
        self.lr_scheduler_info = config.lr_scheduler
        self.metric = getattr(torchmetrics, config.metric)
        self.metric = self.metric()

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func()
        loss = loss(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func()
        loss = loss(logits, y.float())

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
        # config.yaml에서 관리하도록 코드 수정.
        optimizer = getattr(torch.optim, self.optimizer_info.name)
        optimizer = optimizer(
            self.parameters(), lr=self.learning_rate, **self.optimizer_info.args
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_info.name)
        lr_scheduler = lr_scheduler(optimizer, **self.lr_scheduler_info.args)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
