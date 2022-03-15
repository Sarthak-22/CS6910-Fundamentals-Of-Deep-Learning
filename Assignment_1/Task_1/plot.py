import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import function_dataset
from model import function_approximation
import numpy as np
import matplotlib.pyplot as plt

dir = "plots/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def gen_plots(model,name):
    return None


train_dataset = function_dataset("dataset/train.csv")
train_loader = DataLoader(train_dataset)

test_dataset = function_dataset("dataset/test.csv")
test_loader = DataLoader(test_dataset)

valid_dataset = function_dataset("dataset/validation.csv")
valid_loader = DataLoader(valid_dataset)


# Plot of loss variation with epoch
losses = []
with open("losses.txt","r") as f:
    lines = f.readlines()
    for l in lines:
        losses.append(float(l.strip()))

epochs = np.arange(1,len(losses)+1,1)
losses = np.array(losses)
f = plt.figure()
plt.title("Loss v/s epoch")
plt.plot(epochs,losses)
plt.savefig(dir+"loss.png")
plt.close(f)


# Scatter plot of desired output v/s approximated output
model = function_approximation().to(device=device)
model.load_state_dict(torch.load('model_weights.pth'))

desired =[]
approximated = []


for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        out = model(data.float())
       
        approximated.append(out.item())
        desired.append(target.item())
desired = np.array(desired)
approximated = np.array(approximated)

f = plt.figure()
plt.title("Desired v/s Approximated Scatter Plot")

plt.scatter(desired,approximated)
plt.savefig(dir+"scatter.png")
plt.close(f)



for epoch in [1,2,10,50,350]:
    
    model = function_approximation().to(device)
    model.load_state_dict(torch.load(f"epoch{str(epoch)}.pt",map_location=device))
    gen_plots(model,dir+"epoch"+str(epoch))