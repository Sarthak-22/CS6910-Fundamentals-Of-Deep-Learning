import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import function_dataset
from model import function_approximation

data_dir = 'dataset/train.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
learning_rate = 2e-6
epochs = 350
momentum = 0.9

train_dataset = function_dataset(data_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

losses = []

model = function_approximation().to(device=device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


for epoch in range(epochs):
    total_loss = 0
    cnt = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        out = model(data.float())
        loss = criterion(out, target.float())
        cnt+=1
        total_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    losses.append(total_loss/cnt)
    print(f'Epochs:{epoch+1}, Loss:{total_loss/cnt}')
    if(epoch+1 == 1 or epoch+1==2 or epoch+1==10 or epoch+1==50 or epoch+1==epochs):
        torch.save(model.state_dict(), "epoch"+str(epoch+1)+".pt")

with open("losses.txt","w") as f:
    for l in losses:
        f.write(str(l)+"\n")
torch.save(model.state_dict(), 'model_weights.pth')