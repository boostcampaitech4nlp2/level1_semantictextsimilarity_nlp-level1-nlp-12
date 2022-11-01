import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from base import BaseModel
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import paired_cosine_distances

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

class SentTransformer(BaseModel):
    def __init__(self, **configs):
        '''
        ref: 
            1. https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
            2. https://www.sbert.net/examples/training/sts/README.html#training-data
        '''
        super().__init__(**configs) #"sentence-transformers/distiluse-base-multilingual-cased-v2"

        # use `SentenceTransformer` instead of  default huggignface architectures 
        self.lm = None # override self.lm defined in the parent class
        del self.lm
        lm = models.Transformer(self.checkpoint)

        # create pooler for the representations of self.lm 
        pooler = models.Pooling(lm.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

        self.model = SentenceTransformer(modules=[lm, pooler])
        # default `loss_fn` for cosine similarity loss is MSE (ref:https://www.sbert.net/docs/package_reference/losses.html)
        assert self.criterion.__name__ == "mse_loss", "The default fn for calculating cosine similarity is MSE."

        """for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad is False:
                param.requires_grad = True"""

    def forward(self, x):
        '''
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
        https://github.com/UKPLab/sentence-transformers/blob/83eeb5a7b9b81d17a235d76e101cc2912ee1a30d/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py#L14
        '''
        sentences1, sentences2 = [], []
        for sent1, sent2 in zip(x['texts'][0], x['texts'][1]):
            sentences1.append(sent1)
            sentences2.append(sent2)

        embeddings1 = self.model.encode(
            sentences1,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        embeddings2 = self.model.encode(
            sentences2,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        # cosine_scores
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        #dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        return torch.tensor(cosine_scores)

    def training_step(self, batch, batch_idx):
        '''
        loss_fn: https://www.sbert.net/docs/package_reference/losses.html
        '''
        cosine_scores = self.forward(batch)
        targets = batch['label'].to(cosine_scores.device) 
        # mse_loss as designated as default by Sentence-Transformer
        loss = self.criterion(cosine_scores, targets) 
        loss.requires_grad_(True)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        cosine_scores = self.forward(batch)
        targets = batch['label'].to(cosine_scores.device) 
        loss = self.criterion(cosine_scores, targets)
        score = self.metric(cosine_scores, targets)
        self.log("val_loss", loss)
        self.log("val_pearson", score)
        print(f'================batch_idx: {batch_idx} score  {score} ===============')

        return loss

    def test_step(self, batch, batch_idx):
        cosine_scores = self.forward(batch)
        targets = batch['label'].to(cosine_scores.device) 
        score = self.metric(cosine_scores, targets)
        self.log("test_pearson", score)

    def predict_step(self, batch, batch_idx):
        cosine_scores = self.forward(batch)

        return cosine_scores
