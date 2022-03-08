import torch
from torch.utils.data import DataLoader
from model import Image_Classification
from dataset import Image_Dataset

model = Image_Classification()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

train_dataset = Image_Dataset(image_dir='dataset/image_data_dim60.txt', label_dir='dataset/image_data_labels.txt')
train_loader = DataLoader(train_dataset)