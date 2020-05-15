import torch.nn as nn

class BasicModel(nn.Module):

    def set_device(self,
                   device):
        self.device = device

    def get_device(self):
        return self.device

    def set_label_map(self,
                      label_list):
        label_map = {i: label for i, label in enumerate(label_list)}
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map