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


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
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

## MODEL INIT

import torchvision.models as models
import torch.nn as nn

import time

# ---- Step 1: feature extraction (runs the slow backbone ONCE) ----

@torch.inference_mode()
def extract_features(backbone, loader, device):
    backbone.eval()                      # keeps BatchNorm frozen
    feats, labels = [], []
    for i, (X, Y) in enumerate(loader):
        X = X.to(device).contiguous(memory_format=torch.channels_last)
        feats.append(backbone(X).cpu())  # (B, 2048)
        labels.append(Y)
        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)
    return torch.cat(feats), torch.cat(labels)


# ---- Step 2: train the linear head on cached features ----

def train_fc(Xtr, Ytr, Xte, Yte, epochs=20, batch_size=256,
             lr=1e-3, weight_decay=1e-4, save_path="fc_best.pt"):
    clf = nn.Linear(Xtr.shape[1], 2)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        clf.train()
        perm = torch.randperm(len(Xtr))
        total_loss, correct = 0.0, 0

        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            X, Y = Xtr[idx], Ytr[idx]

            logits = clf(X)
            loss = loss_fn(logits, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == Y).sum().item()

        clf.eval()
        with torch.inference_mode():
            test_logits = clf(Xte)
            test_loss = loss_fn(test_logits, Yte).item()
            test_acc = (test_logits.argmax(1) == Yte).float().mean().item()

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss {total_loss/len(Xtr):.4f}, Train Acc {correct/len(Xtr):.4f}, "
              f"Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(clf.state_dict(), save_path)

    print(f"Best test acc: {best_acc:.4f} (saved to {save_path})")
    return clf

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())

    # Backbone only: drop the classifier head, freeze everything
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(device).to(memory_format=torch.channels_last)

    # Deterministic transform (no random flip) for BOTH splits, since features are cached
    train_ds = PrepDataset("data", "train", val_transform)
    test_ds  = PrepDataset("data", "test",  val_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

    if os.path.exists("features.pt"):
        Xtr, Ytr, Xte, Yte = torch.load("features.pt")
        print("Loaded cached features")
    else:
        t0 = time.time()
        print("Extracting train features...")
        Xtr, Ytr = extract_features(backbone, train_loader, device)
        print("Extracting test features...")
        Xte, Yte = extract_features(backbone, test_loader, device)
        torch.save((Xtr, Ytr, Xte, Yte), "features.pt")
        print(f"Extraction took {(time.time()-t0)/60:.1f} min")

    clf = train_fc(Xtr, Ytr, Xte, Yte, epochs=20)