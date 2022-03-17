import torch
from torch import nn
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Linear(in_features=2,out_features=4,bias=True),nn.Tanh())
        self.hidden2 = nn.Sequential( nn.Linear(in_features=4,out_features=2,bias=True),
            nn.Tanh())
        self.output = nn.Sequential(nn.Linear(in_features=2,out_features=1,bias=True),
            nn.Sigmoid())
       


    def forward(self,x,path='all'):
        if(path=='hidden1'):
            return self.hidden1(x)
        if(path=='hidden2'):
            return self.hidden2(x)
        if(path=='output'):
            return self.output(x)
        if(path=='all'):
            return self.output(self.hidden2(self.hidden1(x)))

    

def test():
    print("Testing")
    model = FFNN()
    print(model(torch.tensor([8.85218218,1.47735284])))

if __name__ == "__main__":
    test()