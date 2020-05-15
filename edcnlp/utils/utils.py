from edcnlp.utils.constant import MODELS_dict

def display(option):
    sorted(option.items(), key=lambda s: s[0])
    for k, v in option.items():
        print(k, '=', v)

def build_pretrained_model(option):
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