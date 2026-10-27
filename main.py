# label is like 0 -> 'ai', 1 -> 'real'

'''data_folder_structure:
./
data/
    train/
        ai/
        real/
    test/
        ai/
        real/
'''


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
import os
import random
from PIL import Image


'''
img = Image.open('data/test/ai/0 (2).jpg')

img_arr = torch.tensor(np.array(img))    # 32x32x3

'''

# transform way of doing it is better because it is more efficient and faster

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),                  # upscale 32 -> 224 for ResNet
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                   # HWC uint8 -> CHW float in [0, 1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class PrepDataset(Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        self.samples = []
        for label, cls in enumerate(["real", "ai"]):   # real = 0, ai = 1
            path = os.path.join(root, split, cls)
            for i in os.listdir(path):
                self.samples.append((os.path.join(path, i), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

train_ = PrepDataset("data", "train", train_transform)   # 100000
test_ = PrepDataset("data", "test", val_transform)       # 20000


train_loader = DataLoader(train_, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_, batch_size=64, shuffle=False, num_workers=4)