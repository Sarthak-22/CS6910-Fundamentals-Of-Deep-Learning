import torch
import torch.nn as nn

from model import FFNN
from dataset import Task2aDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FFNN()
model.load_state_dict(torch.load("weights.pt",map_location=device))


trainSet = Task2aDataset("dataset/train_t25.csv")

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=1,
                                          shuffle=True)

devSet = Task2aDataset("dataset/dev_t25.csv")

devLoader = torch.utils.data.DataLoader(devSet, batch_size=1,
                                          shuffle=True)

criterion = nn.BCELoss()
def get_metrics(loader):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total_loss = 0
    cnt = 0
    for i,data in enumerate(loader):
        cnt+=1
        inputs,labels = data
        outputs = model(inputs.to(device))
        labels = labels.unsqueeze(1)
        loss = criterion(outputs,labels)
        
        total_loss+=loss.item()
        if(outputs[0][0] > 0.5):
                if(labels[0][0] == 1.0):
                    TP+=1
                else:
                    FP+=1
        else:
            if(labels[0][0] == 0.0):
                TN+=1
            else:
                FN+=1
    accuracy = (TP+TN)/(FP+FN+TP+TN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)

   
    print("Average Loss: ",total_loss/cnt)
    print("TP:",TP)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)

    print("Accuracy: ",accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score:",2/(1/recall + 1/precision))


print("==Train Set Metrics==")
get_metrics(trainLoader)



print("==Dev Set Metrics==")
get_metrics(devLoader)


