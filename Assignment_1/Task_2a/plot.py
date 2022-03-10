import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from model import FFNN
from matplotlib import cm

from dataset import Task2aDataset

epochs = [360]

def gen_plots(model,name):
    x1 = np.arange(-20,20,0.25,dtype="float32")
    x2 = np.arange(-20,20,0.25,dtype="float32")
    x1,x2 = np.meshgrid(x1,x2)
    
    
    
    hl11 = np.zeros(x1.shape)
    hl12 = np.zeros(x1.shape)
    hl13 = np.zeros(x1.shape)
    hl14 = np.zeros(x1.shape)
    hl21 = np.zeros(x1.shape)
    hl22 = np.zeros(x1.shape)
    y = np.zeros(x1.shape)
    
  
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            
            hl1 = model(torch.tensor([x1[i][j],x2[i][j]]),path="hidden1")
            hl2 = model(hl1,path="hidden2")
            output = model(hl2,path="output")

            #print(hl1)

            hl11[i][j] = hl1[0]
            hl12[i][j]= hl1[1]
            hl13[i][j]= hl1[2]
            hl14[i][j]= hl1[3]

            hl21[i][j]= hl2[0]
            hl22[i][j]= hl2[1]

            y[i][j]= output[0]

   
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl11, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl11.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl12, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl12.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl13, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl13.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl14, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl14.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl21, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl21.png")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, hl22, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_hl22.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.savefig(name+"_output.png")
    plt.close()






        
for epoch in epochs:
    model = FFNN()
    model.load_state_dict(torch.load(f"epoch{str(epoch)}.pt"))
    gen_plots(model,"plots/epoch"+str(epoch))




    

