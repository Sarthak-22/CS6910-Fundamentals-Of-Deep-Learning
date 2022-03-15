import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import train_dataset, val_dataset
from model import Image_Classification, weights_init
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("weight_update", help="Enter the weight update rule", type=str)
args = parser.parse_args()


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
batch_size = 1
learning_rate = 3e-6
epochs = 450
weight_update = args.weight_update


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)



model = Image_Classification().to(device=device)
model.apply(weights_init)

criterion = nn.CrossEntropyLoss()

if weight_update=='generalized_delta':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if weight_update=='delta':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if weight_update=='adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


avg_train_losses = list()
avg_val_losses = list()
for epoch in range(epochs):
    count = 0
    avg_train_loss = 0

    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        count += 1

        image = image.to(device=device)
        label = label.to(device=device)

        out = model(image)
        loss = criterion(out, label.long())

        avg_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    avg_train_loss = avg_train_loss/count
    avg_train_losses.append(avg_train_loss)


    count = 0
    avg_val_loss = 0
    model.eval()
    for batch_idx, (image, label) in enumerate(val_loader):
        count += 1

        image = image.to(device=device)
        label = label.to(device=device)

        out = model(image)
        loss = criterion(out, label.long())

        avg_val_loss += loss.item()
    
    avg_val_loss = avg_val_loss/count
    avg_val_losses.append(avg_val_loss)
    print(f'Epochs:{epoch+1}, Average Train Loss:{avg_train_loss}, Average Validation Loss:{avg_val_loss}')

    if (epoch%10==0):
        if weight_update=='generalized_delta':
            torch.save(model.state_dict(), 'model_weights_generalized_delta.pth')

        if weight_update=='delta':
            torch.save(model.state_dict(), 'model_weights_delta.pth')

        if weight_update=='adam':
            torch.save(model.state_dict(), 'model_weights_adam.pth')




plt.figure()
plt.plot(list(range(1,epochs+1)), avg_train_losses, 'b', label="Train Loss")
plt.plot(list(range(1,epochs+1)), avg_val_losses, 'r', label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title("Average Loss v/s Epoch")
plt.legend()
plt.savefig(f"plots/{weight_update}.png")
plt.show()