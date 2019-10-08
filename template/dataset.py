import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root, transforms=None, num=None):
        super(CelebADataset, self).__init__()

        self.img_root = os.path.join(root, 'img_align_celeba')
        self.attr_root = os.path.join(root, 'Anno/list_attr_celeba.txt')
        self.transforms = transforms

        df = pd.read_csv(self.attr_root, sep='\s+', header=1, index_col=0)
        #print(df.columns.tolist())
        if num is None:
            self.labels = df.values
            self.img_name = df.index.values
        else:
            self.labels = df.values[:num]
            self.img_name = df.index.values[:num]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_root, self.img_name[index]))
        # only use blond_hair, eyeglass, male, smile
        indices = [9, 15, 20, 31]
        label = np.take(self.labels[index], indices)
        label[label==-1] = 0

        if self.transforms is not None:
            img = self.transforms(img)

        return np.asarray(img), label

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dset = CelebADataset('/BS/databases/CelebA')
    loader = DataLoader(dset, batch_size=1, shuffle=True)
    img, label = loader.__iter__().next()
    print(label, label.shape, img.shape)
    plt.imshow(img.squeeze())
    plt.show()
