'''
NER example for CoNLL 2003

'''
import argparse
import random
import numpy as np
import torch
from edcnlp.dataloader.feature import Example
from edcnlp.dataloader.loader import examples_to_dataloader
from edcnlp.model.taskModel import TokenClassification
from edcnlp.utils.utils import display, build_pretrained_model
from edcnlp.utils.trainer import Trainer
import torch.nn.functional as F
from seqeval.metrics import f1_score, classification_report
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--source_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--target_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--train_dir", default='/home/cheny/MyNLP/examples/conll2003ner/en/train.txt', type=str,
                    help="The target language")
parser.add_argument("--dev_dir", default='/home/cheny/MyNLP/examples/conll2003ner/en/dev.txt', type=str,
                    help="The target language")
parser.add_argument("--test_dir", default='/home/cheny/MyNLP/examples/conll2003ner/en/test.txt', type=str,
                    help="The target language")
parser.add_argument("--pretrained_model", default='Bert_base', type=str,
                    help="list:  'MBert_base, Bert_large, Bert_base, Roberta_base, Roberta_large, XLMRoberta_base, XLMRoberta_large")
parser.add_argument("--output_dir", default='save', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_name", default='model', type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--train_batchsize", default=32, type=int)
parser.add_argument("--eval_batchsize", default=32, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dropout_ratio", default=0.4, type=float)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--pos_dim", default=0, type=int)
parser.add_argument("--deprel_dim", default=0, type=int)
parser.add_argument("--ner_dim", default=0, type=int)
parser.add_argument("--freeze", default='0', type=str,
                    help='embedding: freeze embedding, 0: no freeze, n: freeze layers under n')
parser.add_argument("--train_max_seq_length", default=128, type=int)
parser.add_argument("--eval_max_seq_length", default=128, type=int)
parser.add_argument("--train_num_duplicate", default=20, type=int)
parser.add_argument("--eval_num_duplicate", default=20, type=int)
parser.add_argument("--warmup_proportion", default=0.4, type=float)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--crf", default=0, type=int,
                    help="Use CRF = 1, else = 0")



# dataprocessor
class CoNLL2003Processor(object):
    '''Processor for CoNLL-2003 data set.'''

    def __init__(self):
        self.label = ["O",
                      "B-MISC",
                      "I-MISC",
                      "B-PER",
                      "I-PER",
                      "B-ORG",
                      "I-ORG",
                      "B-LOC",
                      "I-LOC"]
        self.create_label_map()

    def get_examples(self,
                     data_dir):
        tsv = self.read_tsv(data_dir)
        examples = self.create_examples(tsv)
        return examples

    def get_labels(self):
        return self.label

    def create_label_map(self):
        self.label_map = {k:  idx for idx, k in enumerate(self.label)}

    def create_examples(self,
                        lines):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            text = sentence
            label = [self.label_map[l] for l in label]
            examples.append(Example(token=text, label=label))
        return examples

    def read_tsv(self,
                 filename):
        '''
        read file
        '''
        print('Reading file: ', filename)
        f = open(filename, encoding='utf-8')
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":

                if len(sentence) > 0:
                    # Add Sentence and label to data
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            # Word
            sentence.append(splits[0])
            # NER Label
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []

        print('Data size: ', len(data))
        return data

def evaluator(model,
              dataloader):
    model.eval()
    device = model.get_device()
    label_map = model.get_label_map()
    data_size = dataloader.size
    y_true = [[] for _ in range(data_size)]
    y_pred = [[] for _ in range(data_size)]
    dev_loss = 0

    for step, batch in enumerate(dataloader.dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_type_ids, attention_mask, valid_ids, pos_ids, ner_ids, deprel_ids, eval_idx, label_ids, label_mask, sen_id = batch

        with torch.no_grad():
            loss, logits = model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             valid_ids=valid_ids,
                             pos_ids=pos_ids,
                             ner_ids=ner_ids,
                             deprel_ids=deprel_ids,
                             labels=label_ids,
                             label_mask=label_mask)
        dev_loss += loss.item()

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        label_mask = label_mask.to('cpu').numpy()
        eval_idx = eval_idx.to('cpu').numpy()
        sen_id = sen_id.to('cpu').numpy()
        for bz_idx, label_i in enumerate(label_ids):
            temp_gold = []
            temp_pred = []
            for tok_idx, tok_i in enumerate(label_i):
                if label_mask[bz_idx][tok_idx] == 0:
                    # stop when label mask = 0
                    s_id = sen_id[bz_idx]
                    y_true[s_id].extend(temp_gold)
                    y_pred[s_id].extend(temp_pred)
                    break
                else:
                    if eval_idx[bz_idx][tok_idx] == 1:
                        # get all prediction when label mask == 1
                        temp_gold.append(label_map[label_ids[bz_idx][tok_idx]])
                        temp_pred.append(label_map[logits[bz_idx][tok_idx]])

    res = f1_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    avg_loss = dev_loss / len(dataloader)
    return res, avg_loss

class Model(nn.Module):

    def __init__(self, pretrained_model, num_labels):
        super().__init__()
        self.encoder = pretrained_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_mask=None, pos_ids=None,
                             ner_ids=None,
                             deprel_ids=None):
        # get bert output
        output = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #last_hidden_states, pooler_output, hidden_states
        sequence_output = output[0]
        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        valid_output = self.valid_word_output(sequence_output, valid_ids)
        # valid_output = [1,1,1,1,0,0,0,0,0,0,0] vs label [1,1,1,1,0,0,0,0,0,0,0] match valid

        # apply dropout and get logits on each time steps
        # label <-> input is 1 to 1 mapping now with some padding, mask out padding later
        sequence_output = self.dropout(valid_output)
        # run through dnn layer to get logits
        logits = self.classifier(sequence_output)
        # get shape of logits
        bz, seq_len, num_label = logits.shape
        # calculate loss
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction='none')
            # reshape loss back to [batchsize, seq_len]
            loss = loss.reshape(bz, seq_len)
            # change dtype to float
            input_mask = label_mask.float()
            # apply mask on time domain so padding on label has loss = 0
            loss = loss * input_mask
            # get avg loss for each example
            loss_batch = loss.mean(-1)
            # get avg loss for minibatch
            avg_loss = loss_batch.mean()
            return avg_loss, logits
        else:
            return logits

    def valid_word_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)

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

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device



if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    option = vars(args)

    print('=' * 30)
    print('Configuration...')
    display(option)

    print('=' * 30)
    print('Building Pretrained Model...')
    Pretrained_model, tokenizer = build_pretrained_model(option)
    # process data
    print('='* 30)
    print('Processing Data...')
    procssor = CoNLL2003Processor()
    label_list = procssor.get_labels()
    train_examples = procssor.get_examples(option['train_dir'])
    dev_examples = procssor.get_examples(option['dev_dir'])
    test_examples = procssor.get_examples(option['test_dir'])
    option['num_labels'] = len(label_list)
    # create dataloader
    # input_ids, token_type_ids, attention_mask, valid_ids, pos_ids, ner_ids, deprel_ids, sen_id
    keys = ['input_ids', 'input_mask', 'token_type_ids', 'valid_idx', 'pos_ids', 'ner_ids', 'deprel_ids', 'eval_idx', 'label_ids', 'label_mask', 'sen_id']

    print('=' * 30)
    print('Building Dataloader...')
    train_dataloader = examples_to_dataloader(train_examples,
                                              option,
                                              tokenizer,
                                              set_type='train',
                                              keys=keys)
    dev_dataloader = examples_to_dataloader(dev_examples,
                                            option,
                                            tokenizer,
                                            set_type='dev',
                                            keys=keys)
    test_dataloader = examples_to_dataloader(test_examples,
                                             option,
                                             tokenizer,
                                             set_type='test',
                                             keys=keys)

    # model
    print('=' * 30)
    print('Building Model...')
    model = TokenClassification(Pretrained_model, option)
    model.set_label_map(label_list)
    model.to(torch.device('cuda:' + option['gpuid']))
    model.set_device('cuda:' + option['gpuid'])

    # trainer
    print('=' * 30)
    print('Training...')
    trainer = Trainer(option=option, model=model, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, evaluator=evaluator)
    trainer.train()
    # test
    print('=' * 30)
    print('Testing...')
    res, _ = evaluator(model, test_dataloader)
    print('Test RES: ', res)