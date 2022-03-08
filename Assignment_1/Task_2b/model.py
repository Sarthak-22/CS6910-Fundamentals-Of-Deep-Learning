import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_Classification(nn.Module):
    def __init__(self):
        super(Image_Classification, self).__init__()

        self.linear1 = nn.Linear(in_features=60, out_features=30)
        self.linear2 = nn.Linear(in_features=30, out_features=15)
        self.linear3 = nn.Linear(in_features=15, out_features=8)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):

        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        #x = self.softmax(self.linear3(x))
        x = F.softmax(self.linear3(x))

        return x

def test():
    model = Image_Classification()
    input = torch.randn([60])
    out = model(input)

    print(model)
    print(out)
    print(torch.argmax(out).float())

#test()