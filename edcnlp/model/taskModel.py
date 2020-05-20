import torch
from torch import nn
import torch.nn.functional as F

from edcnlp.model.inputLayer import InputEmbedding
from edcnlp.model.taskLayer import TokenClassificationLayer, CRFLayer, SequenceClassificationLayer
from edcnlp.model.basicModel import  BasicModel



class TokenClassification(BasicModel):

    def __init__(self,
                 Pretrained_model,
                 option):
        super().__init__()

        self.input_emb = InputEmbedding(Pretrained_model, option)

        #self.input_emb.context_emb.Encoder --> pretrained model
        if option['crf'] == 0:
            self.task_layer = TokenClassificationLayer(option, self.input_emb.input_dim)
        else:
            self.task_layer = CRFLayer(option, self.input_emb.input_dim)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                pos_ids=None,
                ner_ids=None,
                deprel_ids=None,
                #### ADD SRL ID #####
                labels=None,
                label_mask=None):

        input_emb, _ = self.input_emb(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    valid_ids=valid_ids,
                                    pos_ids=pos_ids,
                                    ner_ids=ner_ids,
                                    deprel_ids=deprel_ids)#### ADD SRL ID #####

        if labels is not None:
            loss, logits = self.task_layer(input_emb,
                                            labels=labels,
                                            label_mask=label_mask)
            return loss, logits
        else:
            logits = self.task_layer(input_emb,
                                     label_mask=label_mask)

            return logits

    def set_device(self,
                   device):
        self.device = device
        self.input_emb.set_device(device)

class SequenceClassification(BasicModel):

    def __init__(self,
                 Pretrained_model,
                 option):
        super().__init__()

        self.input_emb = InputEmbedding(Pretrained_model, option)

        #self.input_emb.context_emb.Encoder --> pretrained model
        self.task_layer = SequenceClassificationLayer(option, self.input_emb.input_dim)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                pos_ids=None,
                ner_ids=None,
                deprel_ids=None,
                labels=None,
                label_mask=None):

        _, last_layer_repr = self.input_emb(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    valid_ids=valid_ids,
                                    pos_ids=pos_ids,
                                    ner_ids=ner_ids,
                                    deprel_ids=deprel_ids)

        cls_emb = last_layer_repr[:, 0]

        if labels is not None:
            loss, logits = self.task_layer(cls_emb,
                                            labels=labels,
                                            label_mask=label_mask)
            return loss, logits
        else:
            logits = self.task_layer(cls_emb,
                                     label_mask=label_mask)

            return logits

    def set_device(self,
                   device):
        self.device = device
        self.input_emb.set_device(device)

class MarkerClassification(BasicModel):
    '''
    Use Start Marker for sequence classification
    '''
    def __init__(self,
                 Pretrained_model,
                 option):
        super().__init__()

        self.input_emb = InputEmbedding(Pretrained_model, option)
        # self.input_emb.context_emb.Encoder --> pretrained model
        self.task_layer = SequenceClassificationLayer(option, self.input_emb.input_dim * 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                pos_ids=None,
                ner_ids=None,
                deprel_ids=None,
                labels=None,
                label_mask=None,
                arg1_idx=None,
                arg2_idx=None):

        _, last_layer_repr = self.input_emb(input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            valid_ids=valid_ids,
                                            pos_ids=pos_ids,
                                            ner_ids=ner_ids,
                                            deprel_ids=deprel_ids)
        bz, _, _ = last_layer_repr.size()
        batch_index = [i for i in range(bz)]
        arg1_repr = last_layer_repr[batch_index, arg1_idx]
        arg2_repr = last_layer_repr[batch_index, arg2_idx]

        cls_emb = torch.cat((arg1_repr, arg2_repr), dim=-1)
        if labels is not None:
            loss, logits = self.task_layer(cls_emb,
                                           labels=labels,
                                           label_mask=label_mask)
            return loss, logits
        else:
            logits = self.task_layer(cls_emb,
                                     label_mask=label_mask)

            return logits

    def set_device(self,
                   device):
        self.device = device
        self.input_emb.set_device(device)









