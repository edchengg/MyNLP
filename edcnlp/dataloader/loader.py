import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from edcnlp.dataloader.feature import convert_examples_to_features

class MyDataLoader(object):

    def __init__(self, dataloader,
                 size):
        self.dataloader = dataloader
        self.size = size

    def __len__(self):
        return len(self.dataloader)

def create_dataloader_tagging(features,
                              keys,
                              set_type='train',
                              option=None):
    tensors = []
    for k in keys:
        if k == 'label_mask':
            tensors.append(torch.tensor([f[k] for f in features], dtype=torch.uint8))
        else:
            tensors.append(torch.tensor([f[k] for f in features], dtype=torch.long))


    dataset = TensorDataset(*tensors)

    if set_type == 'train':
        data_sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=data_sampler,
                                batch_size=option['train_batchsize'])
    else:
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=data_sampler,
                                batch_size=option['eval_batchsize'])
    return dataloader


def examples_to_dataloader(examples,
                           option,
                           tokenizer,
                           set_type='train',
                           keys=None):
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            set_type=set_type,
                                            option=option)
    print('Num of instances in {}: {}'.format(set_type, len(features)))
    dataloader = create_dataloader_tagging(features,
                                           keys=keys,
                                           set_type=set_type,
                                           option=option)

    dataloader = MyDataLoader(dataloader, len(examples))
    return dataloader