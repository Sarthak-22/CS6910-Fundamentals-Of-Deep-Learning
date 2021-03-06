# Get outputs for test dataset
import torch
import torch.nn as nn

from model import FFNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FFNN()
model.load_state_dict(torch.load("weights.pt",map_location=device))

lines = []
with open("dataset/test_t25.csv","r") as f:
    lines  = f.readlines()
outputs = []
for line in lines[1:]:
    c = torch.tensor(list(map(float,line.strip().split(",")))[1:])
    print(c)
    outputs.append(model(c).item())

with open("test_output.csv","w") as f:
    f.write(lines[0].strip()+",label\n")
    for i,output in enumerate(outputs):
        if(output>0.5):
            output = 1.0
        else:
            output = 0.0
        f.write(f"{lines[i+1].strip()},{output}\n")