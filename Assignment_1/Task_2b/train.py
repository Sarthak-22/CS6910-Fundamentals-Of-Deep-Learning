import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import Image_Dataset
from model import Image_Classification

train_image_dir = 'dataset/image_data_dim60.txt'
train_label_dir = 'dataset/image_data_labels.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
learning_rate = 1e-3
epochs = 200
#device = 'cpu'

train_dataset = Image_Dataset(image_dir=train_image_dir, label_dir=train_label_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


model = Image_Classification().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device=device)
        label = label.to(device=device)

        out = model(image)
        loss = criterion(out, label.long())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Epochs:{epoch+1}, Loss:{loss}')


torch.save(model.state_dict(), 'model_weights.pth')
