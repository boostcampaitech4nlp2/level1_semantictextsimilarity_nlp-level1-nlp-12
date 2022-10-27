import torch.nn as nn
from abc import abstractmethod
import pytorch_lightning as pl
import transformers


class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    def __init__(
        self,
        checkpoint,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # pretrained model name or checkpoint path
        self.checkpoint = checkpoint 

        # get pretrained model from huggingface hub
        arch = kwargs.pop('architecture', None)
        if arch is None or arch not in [
            'AutoModel', 
            'AutoModelForPreTraining',
            'AutoModelForCausalLM',
            'AutoModelForMaskedLM',
            'AutoModelForSeq2SeqLM',
            'AutoModelForSequenceClassification',
            'AutoModelForMultipleChoice',
            'AutoModelForNextSentencePrediction',
            'AutoModelForTokenClassification',
            'AutoModelForQuestionAnswering'
        ]:
            raise ValueError("The given architecture is None or not available at huggingface.")

        self.lm = getattr(transformers, arch).from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            **kwargs
        )

        # criterion for calculating loss
        self.criterion = criterion

        # metrics
        self.metric_fns = metrics

        # optimizer functions 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        self.log("val_loss", loss)
        for metric_fn in self.metric_fns:
            self.log("val_" + metric_fn.__name__, metric_fn(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        for metric_fn in self.metric_fns:
            self.log("test_" + metric_fn.__name__, metric_fn(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self.forward(x)

        return logits.squeeze()

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.parameters())
        self.hparams['optimizer'] = self.optimizer
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(optimizer=self.optimizer)
            self.hparams['lr_scheduler'] = self.lr_scheduler
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler
            }
        else:
            return self.optimizer

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in model_parameters]) # sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
