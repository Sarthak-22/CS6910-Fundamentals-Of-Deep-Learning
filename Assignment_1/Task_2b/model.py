import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_Classification(nn.Module):
    def __init__(self, l1=256, l2=8):
        super(Image_Classification, self).__init__()

        self.linear1 = nn.Linear(in_features=60, out_features=l1)
        self.linear2 = nn.Linear(in_features=l1, out_features=l2)
        self.linear3 = nn.Linear(in_features=l2, out_features=5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):

        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        #x = self.softmax(self.linear3(x))
        x = self.linear3(x)

        return x

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(768)
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def test():
    model = Image_Classification()
    input = torch.randn([60])
    out = model(input)

    print(model)
    print(out)
    print(torch.argmax(out).float())

#test()