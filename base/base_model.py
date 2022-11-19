from abc import abstractmethod
import transformers
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import model.metric as module_metric
import model.loss as module_loss


class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    def __init__(self, **configs):
        super().__init__()
        self.save_hyperparameters()
        model_args = configs['model_args']
        self.architecture = model_args.pop('architecture', None)

        self.checkpoint = configs['checkpoint']
        self.batch_size = configs['batch_size']
        self.lr = configs.pop('lr', 1e-5)
        self.lr_scheduler = configs.pop('lr_scheduler', None)
        if self.lr_scheduler is not None:
            self.lr_scheduler_args = configs.pop('lr_scheduler_args')
        self.max_epochs = configs.pop('max_epochs')
        self.optimizer = configs['optimizer']
        self.criterion = getattr(module_loss, configs['criterion'])
        self.metric = getattr(module_metric, configs['metric'])

        # get pretrained model from huggingface hub
        if self.architecture is None or self.architecture not in [
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
            raise ValueError(f"The given architecture {self.architecture} is None or not available at huggingface.")

        self.lm = getattr(transformers, self.architecture).from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            **model_args
        )

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        Requires `forward` in the child class 
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
        score = self.metric(logits.squeeze(), y.squeeze())
        self.log("val_loss", loss)
        self.log("val_pearson", score)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        #loss = self.criterion(logits, y.float())
        score = self.metric(logits.squeeze(), y.squeeze())
        #self.log("test_loss", loss)
        self.log("test_pearson", score)
        #return loss


    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self.forward(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)(
            self.parameters(),
            self.lr
        )

        if self.lr_scheduler:
            lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler)(
                    optimizer,
                    **self.lr_scheduler_args
                    )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler
            }
        else:
            return optimizer

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in model_parameters if p is not None]) 
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
