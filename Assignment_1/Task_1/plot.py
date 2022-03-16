import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import function_dataset
from model import function_approximation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

df = pd.read_csv('dataset/func_app1.csv')

dir = "plots/"

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'



train_dataset = function_dataset("dataset/train.csv")
train_loader = DataLoader(train_dataset)

test_dataset = function_dataset("dataset/test.csv")
test_loader = DataLoader(test_dataset)

valid_dataset = function_dataset("dataset/validation.csv")
valid_loader = DataLoader(valid_dataset)


def gen_plots(model,epoch):
    x1 = np.arange(0,6,0.25,dtype="float32")
    x2 = np.arange(0,6,0.25,dtype="float32")
    x1,x2 = np.meshgrid(x1,x2)

    y = np.zeros(x1.shape)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):

            output = model(torch.tensor([x1[i][j],x2[i][j]]))
            y[i][j]= output
    
    f = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x1, x2, y, cmap = cm.jet, linewidth=0, antialiased=False)
    f.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_title(f'Approximated Function after {epoch} Epochs')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.savefig(dir+'epoch'+f'{epoch}'+"_approximated.png")
    plt.show()


# Plot of loss variation with epoch
losses = []
with open("losses.txt","r") as f:
    lines = f.readlines()
    for l in lines:
        losses.append(float(l.strip()))

epochs = np.arange(1,len(losses)+1,1)
losses = np.array(losses)
f = plt.figure()
plt.title("Loss v/s Epoch")
plt.plot(epochs,losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(dir+"loss.png")
plt.show()
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
plt.scatter(desired, approximated, c='b', linewidths=1)
plt.plot(desired, desired, 'r')
plt.xlabel('Desired Function')
plt.ylabel('Approximated Function')
plt.savefig(dir+"scatter.png")
plt.show()
plt.close(f)


f = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_trisurf(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], cmap=cm.jet, linewidth=0, antialiased=False)
f.colorbar(surf, shrink=0.5, aspect=10)
ax.set_title(f'Desired Function')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Desired Function')
plt.savefig(dir+"desired.png")
plt.show()


for epoch in [1,2,10,50,350]:
    model = function_approximation().to(device)
    model.load_state_dict(torch.load(f"epoch{str(epoch)}.pt",map_location=device))
    gen_plots(model,epoch)