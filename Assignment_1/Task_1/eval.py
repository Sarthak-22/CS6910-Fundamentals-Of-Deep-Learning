import torch
from torch.utils.data import DataLoader
from model import function_approximation
from dataset import function_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = function_approximation().to(device=device)
model.load_state_dict(torch.load('model_weights.pth'))

train_dataset = function_dataset()
train_loader = DataLoader(train_dataset)



def accuracy(loader, model):

    print("----------------------------------")
    print("Checking accuracy on training data")
    
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

        print(f"Accuracy of the trained model is {float(num_correct_labels/num_labels)*100:.2f} %")
        print("--------------------------------------------------------------------------------")

    model.train()


accuracy(train_loader, model)