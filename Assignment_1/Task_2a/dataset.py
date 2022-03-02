import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class Task2aDataset(Dataset):
    def __init__(self,data_source):
        
        with open(data_source,"r") as f:
            self.lines = f.readlines()
        self.inputs = []
        self.outputs = []
        for line in self.lines[1:]:
            
            self.linedata = tuple(map(float,line.strip().split(",")))
            self.inputs.append(self.linedata[1:3])
            self.outputs.append(self.linedata[3])

        
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self,idx):
        return self.inputs[idx],self.outputs[idx]

def test():
    print("Testing datset")
    train_dataset = Task2aDataset(data_source="dataset/train_t25.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features)
    
    

if __name__ == '__main__':
    test()

