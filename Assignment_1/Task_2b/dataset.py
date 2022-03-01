from torch.utils.data import Dataset
import numpy as np
import torch


class Image_Dataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image = np.loadtxt(self.image_dir)
        self.label = np.loadtxt(self.label_dir)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img_index = torch.from_numpy(self.image[index])
        label_index = self.label[index]

        return (img_index, label_index)

def test():
    train_dataset = Image_Dataset(image_dir='dataset/image_data_dim60.txt', label_dir='dataset/image_data_labels.txt')
    train_image, train_label = train_dataset[5]

    print(train_image)
    print(train_label)

test()

        

        
