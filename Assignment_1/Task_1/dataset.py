from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class function_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        data = np.array(pd.read_csv(self.data_dir))
        self.input_features = data[:,0:2]
        self.target = data[:,2:]
        self.len = data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        features_index = torch.from_numpy(self.input_features[index])
        target_index = torch.from_numpy(self.target[index])

        return (features_index, target_index)

def test():
    train_dataset = function_dataset(data_dir='dataset/func_app1.csv')
    train_data, train_label = train_dataset[5]
    
    # print(train_data)
    # print(train_label)
    # print(train_data.shape)
    # print(train_label.shape)

    print(train_dataset[:][0][:,0].shape)

#test()
