from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from edcnlp.utils.constant import POS_TO_ID, NER_TO_ID, DEPREL_TO_ID
class Example(object):
    def __init__(self,
                 token=None,
                 label=None,
                 pos=None,
                 ner=None,
                 deprel=None,
                 label_mask=None):
        self.token = token
        self.label = label
        self.pos = ['[PAD]'] * len(token) if pos is None else pos
        self.ner = ['[PAD]'] * len(token) if ner is None else ner
        self.deprel = ['[PAD]'] * len(token) if deprel is None else deprel
        #### ADD SRL ID #####
        self.label_mask = [1] * len(label) if label_mask is None else label_mask


        self.dic = {'token': self.token,
                    'label': self.label,
                    'pos': self.convert_to_id(self.pos, POS_TO_ID),
                    'ner': self.convert_to_id(self.ner, NER_TO_ID),
                    'deprel': self.convert_to_id(self.deprel, DEPREL_TO_ID),
                    #### ADD SRL ID #####
                    'label_mask': self.label_mask
                    }

    def convert_to_id(self, type_list, id):
        ids = [id[t] for t in type_list]
        return ids

    def __getitem__(self, item):
        return self.dic[item]

    def __len__(self):
        return len(self.token)


def convert_examples_to_features(examples,
                                 tokenizer,
                                 set_type,
                                 option):

    if set_type == 'train':
        num_duplicate = option['train_num_duplicate']
        max_seq_length = option['train_max_seq_length']
    else:
        num_duplicate = option['eval_num_duplicate']
        max_seq_length = option['eval_max_seq_length']

    features = []
    for (sen_id, example) in enumerate(examples):
        token_list = example['token']

        if set_type != 'train':
            # change the label mask back to all 1
            example.dic['label_mask'] = [1 for _ in range(len(token_list))]

        # check if exceeds max_seq_length
        total_subwords = []
        num_subwords = [] # record number of subwords for each token
        for token_idx, token in enumerate(token_list):
            subwords = tokenizer.tokenize(token)
            total_subwords.extend(subwords)
            num_subwords.append(len(subwords))

        if len(total_subwords) > max_seq_length - 2:

            # sliding window approach
            start = 0
            end = calculate_end_idx(num_subwords,
                                    start=start,
                                    max_seq_length=max_seq_length)
            while end < len(token_list):
                if start == 0:
                    ignore_idx = 0
                else:
                    ignore_idx = num_duplicate

                feature = prepare_feature(example,
                                          max_seq_length,
                                          tokenizer,
                                          sen_id,
                                          start=start,
                                          end=end,
                                          ignore_idx=ignore_idx)
                features.append(feature)
                start += (end - num_duplicate - start)

                end = calculate_end_idx(num_subwords,
                                        start=start,
                                        max_seq_length=max_seq_length)

            feature = prepare_feature(example,
                                      max_seq_length,
                                      tokenizer,
                                      sen_id,
                                      start=start,
                                      end=len(token_list),
                                      ignore_idx=num_duplicate)
            features.append(feature)
        else:
            start = 0
            end = len(token_list)
            feature = prepare_feature(example,
                                      max_seq_length,
                                      tokenizer,
                                      sen_id = sen_id,
                                      start=start,
                                      end=end)
            features.append(feature)

    return features

def calculate_end_idx(num_subwords,
                      start=0,
                      max_seq_length=128):
    end = 0
    total_subwords = 0
    assert start < len(num_subwords)

    for i in range(start, len(num_subwords)):
        total_subwords += num_subwords[i]
        if total_subwords >= max_seq_length - 2:
            break
        end = i
    return end + 1

def prepare_feature(example,
                    max_seq_length,
                    tokenizer,
                    sen_id,
                    start=0,
                    end=0,
                    ignore_idx=0):

        token_list = example['token'][start:end]
        label_list = example['label'][start:end]
        pos_list = example['pos'][start:end]
        ner_list = example['ner'][start:end]
        #### ADD SRL ID #####
        deprel_list = example['deprel'][start:end]
        label_mask = example['label_mask'][start:end]
        total_subwords = []
        valid_idx = []
        for i, tok in enumerate(token_list):
            subwords = tokenizer.tokenize(tok)
            total_subwords.extend(subwords)
            for m in range(len(subwords)):
                if m == 0:
                    valid_idx.append(1)
                else:
                    valid_idx.append(0)

        total_subwords = total_subwords[:max_seq_length - 2] #cut out long single word such as url link
        valid_idx = valid_idx[:max_seq_length - 2]
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token

        input_token = [bos_token] + total_subwords + [eos_token]
        valid_idx = [0] + valid_idx + [0]
        input_mask = [1] * len(input_token)
        token_type_ids = [0] * len(input_token)
        # pad
        pad_length = max_seq_length - len(input_token)
        input_token += [pad_token] * pad_length
        input_ids = tokenizer.convert_tokens_to_ids(input_token)

        input_mask += [0] * pad_length
        token_type_ids += [0] * pad_length
        valid_idx += [0] * pad_length

        eval_idx = [1] * len(label_list)
        if ignore_idx != 0:
            for ig in range(ignore_idx):
                eval_idx[ig] = 0

        pad_label_length = max_seq_length - len(label_list)

        label_list += [0] * pad_label_length
        label_mask += [0] * pad_label_length
        pos_list += [0] * pad_label_length
        ner_list += [0] * pad_label_length
        deprel_list += [0] * pad_label_length
        #### ADD SRL ID #####
        eval_idx += [0] * pad_label_length

        assert len(input_ids) == max_seq_length
        assert len(valid_idx) == max_seq_length
        assert len(label_list) == max_seq_length
        assert len(label_mask) == max_seq_length

        feature =  {'input_ids':input_ids,
                   'input_mask':input_mask,
                   'token_type_ids':token_type_ids,
                   'valid_idx':valid_idx,
                   'eval_idx':eval_idx,
                   'label_ids':label_list,
                   'label_mask':label_mask,
                   'pos_ids':pos_list,
                   'ner_ids':ner_list,
                   'deprel_ids':deprel_list,
                    #### ADD SRL ID #####
                   'sen_id':sen_id}
        return feature

if __name__ == '__main__':

    examples = [{'token': ['I', 'like', 'apple', '.', 'hello', 'world', '!', 'I', 'like', 'apple', '.', 'hello', 'world', '!'],
                 'label': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
                 'pos':   [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                 'ner':   [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                 'deprel':[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6]}]

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'
    tokenizer.pad_token = '[PAD]'
    option = {'train_num_duplicate': 2, 'train_max_seq_length': 8}
    features = convert_examples_to_features(examples,
                                 tokenizer,
                                 set_type='train',
                                 option=option)
    input_ids = [feature['input_ids'] for feature in features]
    valid_idx = [feature['valid_idx'] for feature in features]
    eval_idx = [feature['eval_idx'] for feature in features]
    word = []
    for ip, val, ev in zip(input_ids, valid_idx, eval_idx):
        tmp = [0] * len(ip)

        ii = 0
        for idx, i in enumerate(val):
            if i == 1:
                tmp[ii] = ip[idx]
                ii += 1
        for idx, i in enumerate(ev):
            if i == 1:
                word.append(tokenizer.convert_ids_to_tokens(tmp[idx]))
    assert word == examples[0]['token']