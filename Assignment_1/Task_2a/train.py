import torch
from dataset import Task2aDataset
from model import FFNN
import torch.optim as optim
import torch.nn  as nn

criterion = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
epochs = 300

trainDataset = Task2aDataset("dataset/train_t25.csv")
trainLoader  = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size,
                                          shuffle=True)
model = FFNN()
model.to(device)

optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.7)



print(device)

for epoch in range(epochs):
    print("Current Epoch: ",epoch)
    total_loss = 0
    cnt = 0
    for i,data in enumerate(trainLoader,0):
        cnt+=1
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Model
        hl1outputs = model(inputs,path='hidden1').to(device)
        hl2outputs = model(hl1outputs,path='hidden2').to(device)
        outputs = model(hl2outputs,path='output')


        labels = labels.unsqueeze(1)
        loss = criterion(outputs,labels)
        total_loss += loss.item()

        

       
        loss.backward()
        optimizer.step()
    print("Epoch Average Loss: ", total_loss/(cnt))

torch.save(model.state_dict(), "weights.pt")











