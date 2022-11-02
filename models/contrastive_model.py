import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel

import pytorch_lightning as pl
from models.loss_function import get_loss_func

from models.optimizer import get_optimizer, get_scheduler


class CustomElectra(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.electra = ElectraModel(config)
        # self.classifier = ElectraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        return sequence_output


class ContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_name = config.model.name

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.lr = config.train.learning_rate

        self.model = CustomElectra.from_pretrained(
            pretrained_model_name_or_path=self.model_name, config=self.model_config
        )
        self.loss_func = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        (
            main_input_ids,
            main_attention_mask,
            pos_input_ids,
            pos_attention_mask,
            neg_input_ids,
            neg_attention_mask,
        ) = batch
        main_logits = self(main_input_ids, main_attention_mask)
        pos_logits = self(pos_input_ids, pos_attention_mask)
        neg_logits = self(neg_input_ids, neg_attention_mask)

        loss = self.loss_func(main_logits, pos_logits, neg_logits)
        self.log("train_triplet_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            main_input_ids,
            main_attention_mask,
            pos_input_ids,
            pos_attention_mask,
            neg_input_ids,
            neg_attention_mask,
        ) = batch
        main_logits = self(main_input_ids, main_attention_mask)
        pos_logits = self(pos_input_ids, pos_attention_mask)
        neg_logits = self(neg_input_ids, neg_attention_mask)

        loss = self.loss_func(main_logits, pos_logits, neg_logits)
        self.log("val_triplet_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)

        return [optimizer], [scheduler]


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.gelu = nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContrastiveElectraForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_name = config.model.name
        self.model_config = AutoConfig.from_pretrained(self.model_name)

        self.electra_model = torch.load("contrastive_trained.pt")
        # self.electra_model = CustomElectra.load_from_checkpoint(config.model.contrastive_ckpt_path)
        self.classifier = ElectraClassificationHead(self.model_config)

        self.loss_func = get_loss_func(config)

    def forward(self, input_ids, attention_mask):
        sequence_output = self.electra_model(input_ids, attention_mask)
        logits = self.classifier(sequence_output)

        return logits


class ContrastiveLearnedElectraModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_name = config.model.name
        self.lr = config.train.learning_rate

        self.model = ContrastiveElectraForSequenceClassification(config)

        self.loss_func = get_loss_func(config)

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        logits = self(input_ids, attention_mask)

        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        logits = self(input_ids, attention_mask)

        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        logits = self(input_ids, attention_mask)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids, attention_mask)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)

        return [optimizer], [scheduler]
