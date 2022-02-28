from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_Classification(nn.Module):
    def __init__(self):
        super(Image_Classification, self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=60, out_features=30)
        self.tanh = nn.Tanh()


    def forward(self, x):

        return x

