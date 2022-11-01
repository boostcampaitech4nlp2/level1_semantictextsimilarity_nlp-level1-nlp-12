import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoModelForSequenceClassification
from models.contrastive_model import ContrastiveElectraForSequenceClassification

from models.optimizer import get_optimizer, get_scheduler
from models.loss_function import get_loss_func


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_name = config.model.name
        self.lr = config.train.learning_rate


        self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name, num_labels=1
            )

        self.loss_func = get_loss_func(config)

    def forward(self, x):
        x = self.model(x)["logits"]

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
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)

        return [optimizer], [scheduler]

    