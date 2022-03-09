import torch
from torch.utils.data import DataLoader
from model import Image_Classification
from dataset import Image_Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Image_Classification().to(device=device)
model.load_state_dict(torch.load('model_weights.pth'))

train_dataset = Image_Dataset(image_dir='dataset/image_data_dim60.txt', label_dir='dataset/image_data_labels.txt')
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
