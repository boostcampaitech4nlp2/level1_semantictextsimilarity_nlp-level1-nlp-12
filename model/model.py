from base import BaseModel
import torch.nn as nn


class Roberta(BaseModel):

    def __init__(
        self,
        checkpoint,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        **kwargs
    ):
        super().__init__(
            checkpoint=checkpoint,
            criterion = criterion,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            **kwargs
        )

    def forward(self, x):
        '''
        Method for task/model-specific forward.
        '''
        x = self.lm(x)['logits']

        return x