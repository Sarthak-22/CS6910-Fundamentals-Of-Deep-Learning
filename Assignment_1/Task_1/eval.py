import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import function_approximation
from dataset import function_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = function_approximation().to(device=device)
model.load_state_dict(torch.load('model_weights.pth'))

train_dataset = function_dataset("dataset/train.csv")
train_loader = DataLoader(train_dataset)

test_dataset = function_dataset("dataset/test.csv")
test_loader = DataLoader(test_dataset)

valid_dataset = function_dataset("dataset/validation.csv")
valid_loader = DataLoader(valid_dataset)


criterion = nn.MSELoss()


def accuracy(loader, model):

    
    
    avg_loss = 0 
    cnt=0

    with torch.no_grad():
        for data, target in loader:
            model.eval()
            data = data.to(device=device)
            target = target.to(device=device)

            out = model(data.float())

            loss = criterion(out,target.float())
            avg_loss += loss
            cnt+=1
        avg_loss = avg_loss/cnt
        print(f"Average loss is: {avg_loss:.2f}")
        print("--------------------------------------------------------------------------------")

   
print("Train Set metrics:")
accuracy(train_loader, model)

print("Test Set metrics:")
accuracy(test_loader, model)

print("Validation Set metrics:")
accuracy(valid_loader, model)