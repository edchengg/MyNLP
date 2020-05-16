from edcnlp.utils.constant import MODELS_dict
import torch
from transformers import BertConfig, BertTokenizer

def display(option):
    sorted(option.items(), key=lambda s: s[0])
    for k, v in option.items():
        print(k, '=', v)

def build_pretrained_model_from_huggingface(option):
    # Define pretrained model
    pretrained_model_dic = MODELS_dict[option['pretrained_model']]
    ckpt = pretrained_model_dic['checkpoint']
    Pretrained_model = pretrained_model_dic['model'].from_pretrained(ckpt, output_hidden_states=True)
    tokenizer = pretrained_model_dic['tokenizer'].from_pretrained(ckpt, do_lower_case=False)
    if 'Bert' in option['pretrained_model']:
        tokenizer.bos_token = '[CLS]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.unk_token = '[UNK]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.cls_token = '[CLS]'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = '[PAD]'
    return Pretrained_model, tokenizer

def build_pretrained_model_from_ckpt(option, device):
    '''
    Loading a pretrained model from a pytorch ckpt
    BERT BASED
    :param option:
    :return:
    '''
    # define BERT model
    pretrained_model_dic = MODELS_dict['Bert_base']
    config = BertConfig.from_json_file(option['pretrained_model'] + '/config.json')
    config.output_hidden_states = True
    Pretrained_model = pretrained_model_dic['model'](config=config)
    Pretrained_model.load_state_dict(torch.load(option['pretrained_model'] + '/pytorch_model.bin', map_location=device),
                                     strict=False)

    if option['pretrained_model'] == 'bibert':
        lower_case_flag = True
    else:
        lower_case_flag = False
    print('lower_case_flag: ', lower_case_flag)
    tokenizer = BertTokenizer.from_pretrained(
         option['pretrained_model'],
        do_lower_case=lower_case_flag)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'
    tokenizer.unk_token = '[UNK]'
    tokenizer.sep_token = '[SEP]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.mask_token = '[MASK]'
    tokenizer.pad_token = '[PAD]'
    return Pretrained_model, tokenizer

