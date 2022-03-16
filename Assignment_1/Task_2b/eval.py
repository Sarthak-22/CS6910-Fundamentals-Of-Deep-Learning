import torch
from torch.utils.data import DataLoader
from model import Image_Classification
from dataset import train_dataset, val_dataset#, test_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
from tuner import config

parser = argparse.ArgumentParser()
parser.add_argument("weight_update", help="Enter the weight update rule", type=str)
args = parser.parse_args()


weight_update = args.weight_update
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

train_loader = DataLoader(dataset=train_dataset)
val_loader = DataLoader(dataset=val_dataset)
#test_loader = DataLoader(dataset=test_dataset)


model = Image_Classification(config["l1"], config["l2"]).to(device=device)

if weight_update=='generalized_delta':
    model.load_state_dict(torch.load('model_weights_generalized_delta.pth'))

if weight_update=='delta':
    model.load_state_dict(torch.load('model_weights_delta.pth'))

if weight_update=='adam':
    model.load_state_dict(torch.load('model_weights_adam.pth'))


def accuracy(loader, model):
    y_pred = []
    y = []

    print("------------------------------------------------------")

    num_correct_labels = 0
    num_labels = 0

    with torch.no_grad():
        for image, label in loader:
            model.eval()
            image = image.to(device=device)
            label = label.to(device=device)

            out = model(image)
            preds = torch.argmax(out)
            y_pred.append(preds.item())
            y.append(int(label.item()))
            num_correct_labels += (preds==label)
            num_labels += 1

        print(f"Accuracy of the model is {float(num_correct_labels/num_labels)*100:.2f} %")
        print("--------------------------------------------------------------------------------")
        print('')

    model.train()

    return (y_pred, y)

    


print("Checking accuracy on train data..")
y_pred_train, y_train = accuracy(train_loader, model)

print("Checking accuracy on validation data..")
y_pred_val, y_val = accuracy(val_loader, model)

# print(len(y_pred_train), len(y_train))
# print(y_pred_train, y_train)

cm_train = confusion_matrix(y_train, y_pred_train)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_train ,display_labels=['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding'])
disp1.plot()
plt.savefig(f"plots/cm_train_{weight_update}.png")
plt.show()


cm_val = confusion_matrix(y_val, y_pred_val)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_val ,display_labels=['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding'])
disp2.plot()
plt.savefig(f"plots/cm_val_{weight_update}.png")
plt.show()


#print("Checking accuracy on test data..")
#accuracy(test_loader, model)


