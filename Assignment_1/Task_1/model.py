import torch
import torch.nn as nn
import torch.nn.functional as F

class function_approximation(nn.Module):
    def __init__(self):
        super(function_approximation, self).__init__()

        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=2)
        self.linear3 = nn.Linear(in_features=2, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):

        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.linear3(x)

        return x

def test():
    model = function_approximation()
    input = torch.Tensor([4.321097794848372, 4.769609253163742])
    out = model(input)
    
    print(input)
    print(model)
    print(out)

#test()