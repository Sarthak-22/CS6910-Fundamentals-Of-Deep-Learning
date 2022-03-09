import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import function_dataset
from model import function_approximation

data_dir = 'dataset/func_app1.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
learning_rate = 1e-6
epochs = 200
#device = 'cpu'

train_dataset = function_dataset(data_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


model = function_approximation().to(device=device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        out = model(data.float())
        loss = criterion(out, target.float())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Epochs:{epoch+1}, Loss:{loss}')


torch.save(model.state_dict(), 'model_weights.pth')