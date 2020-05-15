import torch
from torch import nn
import torch.nn.functional as F
from edcnlp.utils.constant import POS_TO_ID, NER_TO_ID, DEPREL_TO_ID
from edcnlp.model.basicModel import  BasicModel


class CWEmb(BasicModel):
    '''
    Basic Model backbone for contextualize word embedding extraction
    '''

    def __init__(self,
                 Pretrained_model):
        super().__init__()
        self.Encoder = Pretrained_model

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None):

        output = self.Encoder(input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states
        last_layer_repr = output[0]
        # word level embedding
        word_emb = self.valid_word_output(last_layer_repr,
                                          valid_ids)
        return word_emb, last_layer_repr

    def valid_word_output(self,
                          sequence_output,
                          valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,
                                   max_len,
                                   feat_dim,
                                   dtype=torch.float32,
                                   device=self.device)

        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

class InputEmbedding(BasicModel):
    '''
    Input layer embedding
    '''

    def __init__(self, Pretrained_model, option):
        super().__init__()
        self.option = option
        self.context_emb = CWEmb(Pretrained_model)
        self.input_dim =  self.context_emb.Encoder.config.hidden_size + option['pos_dim'] + option['ner_dim'] + option['deprel_dim']
        self.pos_emb = nn.Embedding(len(POS_TO_ID),
                                    option['pos_dim']) if option['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(NER_TO_ID),
                                    option['ner_dim']) if option['ner_dim'] > 0 else None
        self.deprel_emb = nn.Embedding(len(DEPREL_TO_ID),
                                       option['deprel_dim']) if option['deprel_dim'] > 0 else None

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                valid_ids=None,
                pos_ids=None,
                ner_ids=None,
                deprel_ids=None):
        # contextualize embedding
        word_emb, last_layer_repr = self.context_emb(input_ids,
                                                     token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask,
                                                     valid_ids=valid_ids)
        embs = [word_emb]
        if self.option['pos_dim'] > 0:
            embs += [self.pos_emb[pos_ids]]
        if self.option['ner_dim'] > 0:
            embs += [self.ner_emb[ner_ids]]
        if self.option['deprel_dim'] > 0:
            embs += [self.deprel_emb['deprel_ids']]
        embs = torch.cat(embs, dim=2)
        return embs, last_layer_repr

    def set_device(self,
                   device):
        self.device = device
        self.context_emb.set_device(device)










