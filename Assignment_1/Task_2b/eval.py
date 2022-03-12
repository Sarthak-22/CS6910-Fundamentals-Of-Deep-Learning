import torch
from torch.utils.data import DataLoader
from model import Image_Classification
from dataset import train_dataset, val_dataset, test_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("weight_update", help="Enter the weight update rule", type=str)
args = parser.parse_args()


weight_update = args.weight_update
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(dataset=train_dataset)
val_loader = DataLoader(dataset=val_dataset)
test_loader = DataLoader(dataset=test_dataset)


model = Image_Classification().to(device=device)

if weight_update=='generalized_delta':
    model.load_state_dict(torch.load('model_weights_generalized_delta.pth'))

if weight_update=='delta':
    model.load_state_dict(torch.load('model_weights_delta.pth'))

if weight_update=='adam':
    model.load_state_dict(torch.load('model_weights_adam.pth'))



def accuracy(loader, model):

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
            num_correct_labels += (preds==label)
            num_labels += 1

        print(f"Accuracy of the model is {float(num_correct_labels/num_labels)*100:.2f} %")
        print("--------------------------------------------------------------------------------")
        print('')

    model.train()


print("Checking accuracy on train data..")
accuracy(train_loader, model)

print("Checking accuracy on validation data..")
accuracy(val_loader, model)

print("Checking accuracy on test data..")
accuracy(test_loader, model)


