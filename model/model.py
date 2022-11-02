import pytorch_lightning as pl
import transformers
import torch
import torchmetrics
from loss.loss import FocalLoss


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
        self.loss_func = getattr(torch.nn, config.loss)
        self.optimizer_info = config.optimizer
        self.lr_scheduler_info = config.lr_scheduler
        # 평가에 사용할 metric을 호출합니다. metric은 torchmetircs에 존재하는 함수를 지원합니다.
        self.metric = getattr(torchmetrics, config.metric)
        self.metric = self.metric()

    def forward(self, x: torch.Tensor):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_func()
        loss = loss(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx: int):
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

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            self.metric(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx: int):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        """학습에 사용할 optimizer와 lr_scheduler을 반환합니다."""
        optimizer = getattr(torch.optim, self.optimizer_info.name)
        optimizer = optimizer(
            self.parameters(), lr=self.learning_rate, **self.optimizer_info.args
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_info.name)
        lr_scheduler = lr_scheduler(optimizer, **self.lr_scheduler_info.args)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
