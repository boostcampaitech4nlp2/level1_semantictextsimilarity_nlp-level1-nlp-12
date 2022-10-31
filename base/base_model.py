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
        self.lr = configs.pop('lr', 1e-5)
        self.lr_scheduler = configs.pop('lr_scheduler', None)
        if self.lr_scheduler is not None:
            self.lr_scheduler_args = configs.pop('lr_scheduler_args')
        self.max_epochs = configs.pop('max_epochs')
        self.optimizer = configs['optimizer']
        #self.criterion = configs['criterion']
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

        :return: Model output
        """
        pass

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([p.numel() for p in model_parameters]) 
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
