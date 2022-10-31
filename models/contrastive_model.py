import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel

import pytorch_lightning as pl

from models.optimizer import get_optimizer, get_scheduler
from models.loss_function import get_loss_func, TripletLoss


class CustomElectraForSequenceClassification(ElectraPreTrainedModel):
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
        # 여기 아래서부터 문제 ##
        print("🤢🤢")
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        print("🤢🤢🤢")
        sequence_output = discriminator_hidden_states[0]

        return sequence_output


class ContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        
        self.config = config
        self.model_name = config.model.name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.lr = config.train.learning_rate

        self.model = CustomElectraForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, config=self.model_config
        )
        self.loss_func = TripletLoss(margin=2)

    def forward(self, main_input_ids, pos_input_ids, neg_input_ids):
        print("😡😡😡Checking😡😡😡" )
        # 😡😡😡 여기 아래부터 문제 😡😡😡
        main_logits = self.model(main_input_ids)
        print(main_logits)
        pos_logits = self.model(pos_input_ids)
        neg_logits = self.model(neg_input_ids)

        return main_logits, pos_logits, neg_logits

    def training_step(self, batch):
        main, pos, neg= batch
        main_logits, pos_logits, neg_logits = self(main, pos, neg)

        loss = self.loss_func(main_logits, pos_logits, neg_logits)

        return loss
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)

        return [optimizer], [scheduler]
        