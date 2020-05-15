from transformers import BertModel,BertTokenizer,\
    XLMModel, XLMTokenizer, \
    RobertaModel, RobertaTokenizer,\
    XLMRobertaModel, XLMRobertaTokenizer


MODELS_dict = {
          'MBert_base': {'model': BertModel, 'tokenizer': BertTokenizer, 'checkpoint': 'bert-base-multilingual-cased'},
          'Bert_large': {'model': BertModel, 'tokenizer': BertTokenizer, 'checkpoint': 'bert-large-cased'},
          'Bert_base': {'model': BertModel, 'tokenizer': BertTokenizer, 'checkpoint': 'bert-base-cased'},
          "Roberta_base": {'model': RobertaModel, 'tokenizer': RobertaTokenizer, 'checkpoint': 'roberta-base'},
          "Roberta_large": {'model': RobertaModel, 'tokenizer': RobertaTokenizer, 'checkpoint': 'roberta-large'},
          "XLMRoberta_base": {'model': XLMRobertaModel, 'tokenizer': XLMRobertaTokenizer, 'checkpoint': 'xlm-roberta-base'},
          "XLMRoberta_large": {'model': XLMRobertaModel, 'tokenizer': XLMRobertaTokenizer, 'checkpoint': 'xlm-roberta-large'}
          }

_PAD = '[PAD]'
# Stanford dependency relation type 38
DEPREL_TYPE = [_PAD, 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
          'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj',
          'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed',
          'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj',
          'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct',
          'reparandum', 'root', 'vocative', 'xcomp', 'ref'
          ]

DEPREL_TO_ID = {k: idx for idx, k in enumerate(DEPREL_TYPE)}
# UD pos tag type 17
POS_TYPE = [_PAD, "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
POS_TO_ID = {k: idx for idx, k in enumerate(POS_TYPE)}

NER_TYPE = [_PAD, 'PER', 'ORG', 'LOC', 'MISC']
NER_TO_ID = {k: idx for idx, k in enumerate(NER_TYPE)}

