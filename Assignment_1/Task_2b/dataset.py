from torch.functional import norm
from torch.utils import data
from torch.utils.data import Dataset, random_split
import numpy as np
import torch
import pandas as pd

train_image_dir = 'dataset/image_data_dim60.txt'
train_label_dir = 'dataset/image_data_labels.txt'

class Image_Dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image = np.loadtxt(self.image_dir)
        self.label = np.loadtxt(self.label_dir)

        mask = (self.label==0) | (self.label==1) | (self.label==5) | (self.label==6) | (self.label==7)
        self.label, self.image = self.label[mask], self.image[mask]
        self.label[(self.label==5)], self.label[(self.label==6)], self.label[(self.label==7)] = 2, 3, 4 

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_index = torch.from_numpy(self.image[index].astype(np.float32))
        label_index = torch.tensor(self.label[index])

        return (img_index, label_index)


dataset = Image_Dataset(image_dir=train_image_dir, label_dir=train_label_dir)

# 70-20-10 train-dev-test split
# train_size = int(0.7*len(dataset))
# val_size = int(0.2*len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_size = int(0.8*len(dataset))
val_size =  len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# scaler = StandardScaler()
# scaler.fit(train_dataset[:][0])

# train_dataset[:][0] = torch.tensor(scaler.transform(train_dataset[:][0].item()))
# val_dataset[:][0] = torch.tensor(scaler.transform(val_dataset[:][0].item()))
# test_dataset[:][0] = torch.tensor(scaler.transform(test_dataset[:][0].item()))





def test():
    dataset = Image_Dataset(image_dir='dataset/image_data_dim60.txt', label_dir='dataset/image_data_labels.txt')
    train_image, train_label = dataset[567]
    
    print(dataset[:][0].shape, dataset[:][1].shape)
    print(pd.DataFrame(dataset[:][1]).value_counts())
    print(train_image.shape)
    print(train_label)
    

#test()

        