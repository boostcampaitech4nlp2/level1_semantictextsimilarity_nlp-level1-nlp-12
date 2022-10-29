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
        self.save_hyperparameters(ignore='metrics')
        model_args = configs['model_args']
        self.architecture = model_args.pop('architecture', None)

        self.checkpoint = configs['checkpoint']
        self.lr = configs.pop('lr', 1e-5)
        self.lr_scheduler = configs.pop('lr_scheduler', None)
        if self.lr_scheduler is not None:
            self.lr_scheduler_args = configs.pop('lr_scheduler_args')
        self.max_epochs = configs.pop('max_epochs')
        self.optimizer = configs['optimizer']
        self.criterion = configs['criterion']
        self.criterion = getattr(module_loss, self.criterion)
        self.metrics = configs.pop('metrics')

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
            raise ValueError("The given architecture is None or not available at huggingface.")

        self.lm = getattr(transformers, self.architecture).from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            **model_args
        )

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
        for metric in self.metrics:
            score = getattr(module_metric, metric)(logits.squeeze(), y.squeeze())
            print('val_'+metric, score)
            self.log("val_" + metric, score)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        for metric in self.metrics:
            score = getattr(module_metric, metric)(logits.squeeze(), y.squeeze())
            self.log("test_" + metric, score)

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
        params = sum([p.numel() for p in model_parameters]) # sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
