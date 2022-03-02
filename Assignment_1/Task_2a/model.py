import torch
from torch import nn
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=2,out_features=4,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=4,out_features=2,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=2,out_features=1,bias=True),
            nn.Sigmoid(),

        )
    def forward(self,x):
        return self.model(x)

def test():
    print("Testing")
    model = FFNN()
    print(model(torch.tensor([8.85218218,1.47735284])))

if __name__ == "__main__":
    test()