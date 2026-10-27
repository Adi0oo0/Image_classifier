# label is like 0 -> 'real', 1 -> 'ai'

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

device = torch.device("cpu")


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


train_loader = DataLoader(train_, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_, batch_size=32, shuffle=False, num_workers=4)


## MODEL INIT

import torchvision.models as models
import torch.nn as nn

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 2)   # 2 classes
print("Model loaded with pretrained weights from ImageNet")
model.to(device)


## FOR TRAINING ONLY THE LAST LAYER

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

## LOSS AND OPTIMIZER

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

def train_loop(model, loader, training: bool):
    model.fc.train() if training else model.eval()

    total_loss = 0.0
    total, correct = 0, 0 

    with torch.set_grad_enabled(training):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
    
            # Forward pass
            logits = model(X)
            loss = loss_fn(logits, Y)
    
            # Backward pass and optimization
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == Y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


if __name__ == "__main__":
    for epoch in range(2):
        train_loss, train_acc = train_loop(model, train_loader, training=True)
        test_loss, test_acc = train_loop(model, test_loader, training=False)
        torch.save(model.fc.state_dict(), "fc.pt")

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")