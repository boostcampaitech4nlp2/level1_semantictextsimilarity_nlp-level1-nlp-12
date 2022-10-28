import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Roberta(BaseModel):
    def __init__(self, **configs):
        super().__init__(**configs)

    def forward(self, x):
        '''
        Method for task/model-specific forward.
        '''
        x = self.lm(x)['logits']

        return x
