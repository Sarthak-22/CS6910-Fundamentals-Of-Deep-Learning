import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import FFNN 



class Task2aDataset(Dataset):
    def __init__(self,data_source):
        
        with open(data_source,"r") as f:
            self.lines = f.readlines()
        self.inputs = []
        self.outputs = []
        for line in self.lines[1:]:
            
            self.linedata = tuple(map(float,line.strip().split(",")))
            self.inputs.append(torch.tensor(self.linedata[1:3]))
            self.outputs.append(torch.tensor(self.linedata[3]))

        
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self,idx):
        return self.inputs[idx],self.outputs[idx]

def test():
    print("Testing datset")
    train_dataset = Task2aDataset(data_source="dataset/train_t25.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    #train_features, train_labels = next(iter(train_dataloader))
    for i,data in enumerate(train_dataloader,0):
        if(i==0):
            inputs,labels = data
            model = FFNN()
            print(model(inputs))

    
    

if __name__ == '__main__':
    test()

