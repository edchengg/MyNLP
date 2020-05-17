'''
SRL example of OntoNote 5.0

'''
import argparse
import random
import numpy as np
import torch
from edcnlp.dataloader.feature import Example
from edcnlp.dataloader.loader import examples_to_dataloader
from edcnlp.model.taskModel import TokenClassification
from edcnlp.utils.utils import display, build_pretrained_model_from_huggingface
from edcnlp.utils.trainer import Trainer
from edcnlp.utils.ontonote import SrlProcessor
from seqeval.metrics import f1_score, classification_report
import torch.nn.functional as F

# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--source_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--target_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--train_dir", default='/home/cheny/MyNLP/examples/ontonote5.0srl/conll-formatted-ontonotes-5.0/data/train/', type=str,
                    help="The target language")
parser.add_argument("--dev_dir", default='/home/cheny/MyNLP/examples/ontonote5.0srl/conll-formatted-ontonotes-5.0/data/development/', type=str,
                    help="The target language")
parser.add_argument("--test_dir", default='/home/cheny/MyNLP/examples/ontonote5.0srl/conll-formatted-ontonotes-5.0/data/test/', type=str,
                    help="The target language")
parser.add_argument("--pretrained_model", default='Bert_base', type=str,
                    help="list:  'MBert_base, Bert_large, Bert_base, Roberta_base, Roberta_large, XLMRoberta_base, XLMRoberta_large")
parser.add_argument("--output_dir", default='save', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_name", default='model', type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--train_batchsize", default=32, type=int)
parser.add_argument("--eval_batchsize", default=128, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=15, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dropout_ratio", default=0.1, type=float)
parser.add_argument("--gpuid", default='1', type=str)
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
    # remove 'B-V' --> 'O'
    for sen_idx in range(len(y_true)):
        for tag_idx in range(len(y_true[sen_idx])):
            if y_true[sen_idx][tag_idx] == 'B-V':
                y_true[sen_idx][tag_idx] = 'O'
            if y_pred[sen_idx][tag_idx] == 'B-V':
                y_pred[sen_idx][tag_idx] = 'O'

    res = f1_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    avg_loss = dev_loss / len(dataloader)
    return res, avg_loss


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    option = vars(args)

    print('=' * 30)
    print('Configuration...')
    display(option)

    processor = SrlProcessor()
    print('=' * 30)
    print('Building Pretrained Model...')
    Pretrained_model, tokenizer = build_pretrained_model_from_huggingface(option, add_tokens=[processor.START_MARKER, processor.END_MARKER])
    # process data
    print('='* 30)
    print('Processing Data...')
    label_list = processor.get_labels()
    train_examples = processor.get_examples(option['train_dir'])
    dev_examples = processor.get_examples(option['dev_dir'])
    test_examples = processor.get_examples(option['test_dir'])
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
    trainer = Trainer(option=option,
                      model=model,
                      train_dataloader=train_dataloader,
                      dev_dataloader=dev_dataloader,
                      evaluator=evaluator)
    trainer.train()
    # test
    print('=' * 30)
    print('Testing...')
    res, _ = evaluator(model, test_dataloader)
    print('Test RES: ', res)