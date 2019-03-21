from __future__ import print_function
from PIL import Image
import os
import sys
import numpy as np
import argparse

import torch.utils.data as data


class Fruit(data.Dataset):
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.train=train

        if (self.train):
            self.data = np.load(os.path.join(self.root_dir, "train_data.npy"))
            self.labels = np.load(os.path.join(self.root_dir, "train_labels.npy"))
        else:
            self.data = np.load(os.path.join(self.root_dir, "validation_data.npy"))
            self.labels = np.load(os.path.join(self.root_dir, "validation_labels.npy"))

        self.data = self.data.transpose((0, 2, 3, 1))
    
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        #img = Image.fromarray(img.astype('uint8'))
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return (len(self.data))

     
