import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base import BaseModel


class Base(BaseModel):
    '''
    baseline 그대로의 모델
    '''
    def __init__(self, **configs):
        super().__init__(**configs)

    def forward(self, x):
        '''
        Method for task/model-specific forward.
        '''
        x = self.lm(x)['logits']

        return x

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

class ElectraDropout(BaseModel):
    def __init__(self, **configs):
        super().__init__(**configs)
        F.dropout()
        for name, layer in self.lm.named_children():
            if 'Embedding' in name or 'embedding' in name:
                ###################
                print('Applied dropout at the output of embedidng layer')

    def forward(self, x):
        '''
        Method for task/model-specific forward.
        '''
        x = self.lm(x)['logits']

        return x
