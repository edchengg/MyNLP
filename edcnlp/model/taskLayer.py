import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF

class TokenClassificationLayer(nn.Module):
    '''
    task-specific layer for token-level classification
    '''
    def __init__(self,
                 option,
                 input_dim=768):
        super().__init__()
        self.num_labels = option['num_labels']
        self.dropout = nn.Dropout(option['dropout_ratio'])
        self.projection_layer = nn.Linear(input_dim, self.num_labels)

    def forward(self,
                input_emb,
                labels=None,
                label_mask=None):
        input_emb = self.dropout(input_emb)
        logits = self.projection_layer(input_emb)
        if labels is not None:
            bz, seq_len, _ = logits.shape
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction='none')
            # (batchsize, seq_len)
            loss = loss.reshape(bz, seq_len)
            loss = self.apply_label_mask(loss, label_mask)
            return loss, logits
        else:
            return logits

    def apply_label_mask(self,
                         loss,
                         label_mask):
        loss = loss * label_mask.float() # label mask
        loss = loss.mean(-1) # average loss in each sentence
        loss = loss.mean() # average loss in mini batch
        return loss

class CRFLayer(nn.Module):
    '''
    task-specific layer for conditional random field structure prediction
    '''
    def __init__(self,
                 option,
                 input_dim=768):
        super().__init__()
        self.num_labels = option['num_labels']
        self.dropout = nn.Dropout(option['dropout_ratio'])
        self.projection_layer = nn.Linear(input_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self,
                input_emb,
                labels=None,
                label_mask=None):
        input_emb = self.dropout(input_emb)
        logits = self.projection_layer(input_emb)
        if labels is not None:
            loss = self.crf(logits, labels, mask=label_mask, reduction='mean') # log likelihood
            return -loss, logits
        else:
            out = self.crf.decode(logits, mask=label_mask)
            return out

class SequenceClassificationLayer(nn.Module):
    '''
    task-specific layer for sequence classification
    '''
    def __init__(self,
                 option,
                 input_dim=768):
        super().__init__()
        self.num_labels = option['num_labels']
        self.dropout = nn.Dropout(option['dropout_ratio'])
        self.projection_layer = nn.Linear(input_dim, self.num_labels)

    def forward(self,
                cls_emb,
                labels=None,
                label_mask=None):
        # [bz, input_dim]
        cls_emb = self.dropout(cls_emb)
        logits = self.projection_layer(cls_emb)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        else:
            return logits

