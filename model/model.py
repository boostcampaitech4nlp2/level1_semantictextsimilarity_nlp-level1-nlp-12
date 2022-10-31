import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
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

    

class SentTransformer(pl.LightningModule):
    
    def __init__(self, **configs):
        '''
        ref: 
            1. https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
            2. https://www.sbert.net/examples/training/sts/README.html#training-data
        '''
        from sentence_transformers import SentenceTransformer, models, losses
        super().__init__(**self) #"sentence-transformers/distiluse-base-multilingual-cased-v2"

        # get another pretrained model for reference sentences

        # create pooler for the representations of self.lm 
        self.pooler = nn.Linear(self.lm.config.dim, )

        self.model = SentenceTransformer(modules=[self.lm, self.pooler])

    def forward(self, x):
        x = self.model()

    

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
