import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def normalize_pc_pair(input, gt):
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:,0:3], axis=0, keepdims=True)
    gt[:,0:3] = gt[:,0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:,0:3] ** 2, axis=-1)),axis=0,keepdims=True)
    gt[ :, 0:3] = gt[:,0:3] / np.expand_dims(furthest_distance,axis=-1)
    input[ :, 0:3] = input[ :, 0:3] - centroid
    input[ :, 0:3] = input[ :, 0:3] / np.expand_dims(furthest_distance,axis=-1)
    
    return input, gt

class PointDetailDataset(Dataset):
    def __init__(self, root_dir, train=True , transform=None):
        if(train):
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            raise NotImplementedError("Test set not implemented")
        self.transform = transform
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.files[idx]
        data = np.load(os.path.join(self.root_dir, file))

        indices = np.random.choice(data.shape[0], 2048, replace=False)
        y = data[indices].astype(np.float32)

        mean = 0
        std = 0.01
        noise = np.random.normal(mean, std, y.shape).astype(np.float32)

        x = y + noise

        print(x.dtype)
        print(y.dtype)

        if self.transform:
            data = self.transform(data)

        return (x, y)