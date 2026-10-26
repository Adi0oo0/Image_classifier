# Build an AI-vs-Real Image Detector (ResNet18, free Colab T4)

A beginner-friendly, copy-paste guide. You will fine-tune a small pretrained ResNet18 to say "REAL" or "AI-generated", then export it so it runs on your laptop CPU.

**What you need:** a Google account (Colab + Drive) and a free Kaggle account. No GPU on your laptop is required.

---

## The big picture

| Step | What you do | Time |
|---|---|---|
| 0 | Understand the plan and its limits | 5 min |
| 1 | Set up Colab + Google Drive | 10 min |
| 2 | Download the dataset | 10 min |
| 3 | Load data and build the model | 10 min |
| 4 | Train the head (quick first result) | 15-30 min |
| 5 | Fine-tune the whole model (can span sessions) | 1-2 hours total |
| 6 | Test the model honestly | 5 min |
| 7 | Export to ONNX | 5 min |
| 8 | Run it on your laptop | 10 min |
| 9 | Make it better (round 2) | later |

---

## Step 0: Read this first

**About SynthID:** it is Google's private watermark. Only Google's detector can read it, so you cannot use it in your own free model. The free open standard is **C2PA / Content Credentials** (metadata attached to images by some tools and cameras). It's handy as an optional extra check (see Step 10), but most images online have that metadata stripped, so your model has to look at the pixels. That is what this project does.

**Realistic expectations:**
- On images similar to your training data, you may see 90%+ accuracy.
- On images from AI generators your model never saw, accuracy drops, often to 70-85%. That's normal for this problem, not a bug.
- The best fix is more varied training data (Step 9).

**Why ResNet18?** About 11M parameters, ~45 MB exported, trains fast on a T4 and runs well on a laptop CPU.

---

## Step 1: Set up Colab and Drive

1. Go to https://colab.research.google.com and click **New notebook**.
2. Menu: **Runtime → Change runtime type → T4 GPU → Save**.
3. Name the notebook `ai_detector.ipynb` (top left).

**Cell 1: check the GPU**

```python
!nvidia-smi
```

You should see "Tesla T4". If not, redo the runtime change.

**Cell 2: connect Google Drive** (this is where checkpoints are saved so nothing is lost when the session ends)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = "/content/drive/MyDrive/ai_detector"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Saving to:", SAVE_DIR)
```

Click "Allow" when asked.

> **Every new session**, run Cell 1 and Cell 2 again first. Colab forgets everything except what is in Drive.

---

## Step 2: Get the dataset

We start with **CIFAKE** because it is small and easy. Its images are only 32×32, so it is *not* realistic, but it's perfect for learning the whole pipeline. (Step 9 upgrades to better data.)

### 2a. Get a Kaggle API key (one time)

1. Log in at https://www.kaggle.com → click your profile picture → **Settings**.
2. Scroll to **API** → **Create New Token**. A file `kaggle.json` downloads.
3. In Colab, click the folder icon on the left, then the upload icon, and upload `kaggle.json` to `/content`.

### 2b. Download and unzip

```python
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# Download the zip into Drive once, so later sessions can reuse it
!kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p /content/drive/MyDrive/ai_detector
```

Then unzip to Colab's **local disk** (much faster to read than Drive):

```python
!mkdir -p /content/data
!unzip -q /content/drive/MyDrive/ai_detector/cifake-real-and-ai-generated-synthetic-images.zip -d /content/data
!ls /content/data
```

You should see `train` and `test`, each containing `REAL` and `FAKE` folders.
(If the dataset name has changed or is unavailable, search "CIFAKE" on Kaggle or Hugging Face and adjust. Always check the license.)

> **Every new session**, re-run only the `unzip` cell (the zip is already in Drive).

---

## Step 3: Load data and build the model

**Cell: imports and settings**

```python
import os, json, random, io
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageFilter

device = "cuda"
DATA_DIR   = "/content/data"
IMG_SIZE   = 224
BATCH_SIZE = 64          # if you get "CUDA out of memory", change to 32
MAX_TRAIN  = 30000       # use a subset to keep epochs fast; set None for all 100k
```

**Cell: augmentations.** These make the model robust to real-world images (compressed, blurred), which is why we add them.

```python
class RandomJPEG:
    """Randomly re-compress the image like social media would."""
    def __init__(self, p=0.5, qmin=60, qmax=95):
        self.p, self.qmin, self.qmax = p, qmin, qmax
    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    RandomJPEG(), RandomBlur(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

> We deliberately do **not** use color jitter or heavy distortions. They can destroy the subtle traces we're trying to detect.

**Cell: datasets and loaders**

```python
train_full = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_full   = datasets.ImageFolder(f"{DATA_DIR}/train", transform=eval_tf)   # same files, no augmentation
test_set   = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=eval_tf)

CLASSES = train_full.classes          # alphabetical: ['FAKE', 'REAL'] -> FAKE=0, REAL=1
print("Classes:", CLASSES)
with open(f"{SAVE_DIR}/classes.json", "w") as f:
    json.dump(CLASSES, f)

# Split train into 90% train / 10% validation, and optionally shrink it
random.seed(42)
idx = list(range(len(train_full)))
random.shuffle(idx)
n_val = int(0.1 * len(idx))
val_idx, train_idx = idx[:n_val], idx[n_val:]
if MAX_TRAIN:
    train_idx = train_idx[:MAX_TRAIN]
    val_idx   = val_idx[:MAX_TRAIN // 10]

train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(Subset(val_full, val_idx), batch_size=BATCH_SIZE,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE,
                          num_workers=2, pin_memory=True)

print(len(train_idx), "train |", len(val_idx), "val |", len(test_set), "test")
```

**Cell: the model**

```python
def build_model():
    m = torchvision.models.resnet18(weights="DEFAULT")   # pretrained on ImageNet
    m.fc = nn.Linear(m.fc.in_features, 2)                # new head: FAKE / REAL
    return m.to(device)

def set_freeze(model, freeze_backbone):
    for p in model.parameters():
        p.requires_grad = not freeze_backbone
    for p in model.fc.parameters():
        p.requires_grad = True        # the head always trains
```

> Want ResNet50 later? Change `resnet18` to `resnet50` in `build_model`. Nothing else changes.

---

## Step 4: The training function

One function does everything: trains, validates, and **saves a checkpoint after every epoch** so you can resume after a disconnect.

```python
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.autocast("cuda"):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def run_training(name, epochs, lr, freeze_backbone, init_from=None):
    ckpt = f"{SAVE_DIR}/{name}_last.pt"
    best_path = f"{SAVE_DIR}/{name}_best.pt"
    model = build_model()
    state = None
    start, best = 0, 0.0

    if os.path.exists(ckpt):                       # resume after a disconnect
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"Resuming '{name}' from epoch {start}")
    elif init_from and os.path.exists(init_from):  # start from an earlier phase
        model.load_state_dict(torch.load(init_from, map_location=device)["model"])
        print("Starting from", init_from)

    set_freeze(model, freeze_backbone)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    if state is not None:
        opt.load_state_dict(state["opt"])
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(start, epochs):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if i % 100 == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_loader)} loss {loss.item():.4f}")

        val_acc = evaluate(model, val_loader)
        print(f"[{name}] epoch {epoch} | train loss {running/len(train_loader):.4f} | val acc {val_acc:.4f}")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": max(best, val_acc)}, ckpt)
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict()}, best_path)
            print("  saved new best model")
    return best_path
```

### Phase A: train only the head (quick win)

```python
head_best = run_training("head", epochs=3, lr=1e-3, freeze_backbone=True)
```

Expect validation accuracy to climb above ~80% within a few epochs.

### Phase B: fine-tune everything

Use a **low** learning rate so you don't wreck the pretrained knowledge.

```python
full_best = run_training("full", epochs=8, lr=1e-4,
                         freeze_backbone=False,
                         init_from=f"{SAVE_DIR}/head_best.pt")
```

**If the session disconnects:** reconnect, re-run Steps 1-3 cells, then re-run the *same* `run_training(...)` cell. It resumes from the last finished epoch automatically.

**Tips:**
- Each epoch prints progress every 100 batches. If one epoch takes more than ~10 minutes, lower `MAX_TRAIN` (for example 15000).
- If validation accuracy stops improving for 2-3 epochs, stop. More epochs won't help.
- Optional: after a few epochs, run another round with `lr=1e-5` for 2 epochs to polish.

---

## Step 5: Test honestly

The test set was never used in training or model selection, so this number is the honest one.

```python
from sklearn.metrics import classification_report, confusion_matrix

model = build_model()
model.load_state_dict(torch.load(f"{SAVE_DIR}/full_best.pt", map_location=device)["model"])
model.eval()

preds, labels = [], []
with torch.no_grad(), torch.autocast("cuda"):
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()

print(classification_report(labels, preds, target_names=CLASSES))
print(confusion_matrix(labels, preds))
```

How to read it:
- **Precision (FAKE)**: when it says "AI", how often is it right?
- **Recall (FAKE)**: of all the AI images, how many did it catch?
- The confusion matrix shows which mistakes it makes (real called fake, or fake called real).

---

## Step 6: Export to ONNX (small, laptop-friendly)

```python
!pip install -q onnx onnxscript

model = build_model()
model.load_state_dict(torch.load(f"{SAVE_DIR}/full_best.pt", map_location=device)["model"])
model.eval().cpu()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(
    model, dummy, f"{SAVE_DIR}/detector.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)
print("Exported. Size (MB):", os.path.getsize(f"{SAVE_DIR}/detector.onnx") / 1e6)
```

If you get an error about `dynamo`, delete `dynamo=False,` and run it again.

Download `detector.onnx` and `classes.json` from your Drive folder (`MyDrive/ai_detector`) to your laptop.

---

## Step 7: Run it on your laptop

On your laptop, install Python 3.10+, then:

```bash
pip install onnxruntime pillow numpy
```

Create `detect.py` in the same folder as `detector.onnx` and `classes.json`:

```python
import sys, json
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

classes = json.load(open("classes.json"))            # e.g. ["FAKE", "REAL"]
session = ort.InferenceSession("detector.onnx", providers=["CPUExecutionProvider"])

img = Image.open(sys.argv[1]).convert("RGB").resize((224, 224))
x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
x = x.transpose(2, 0, 1)[None].astype(np.float32)    # shape (1, 3, 224, 224)

logits = session.run(None, {"image": x})[0][0]
probs = np.exp(logits - logits.max()); probs /= probs.sum()

i = int(probs.argmax())
label = "AI-generated" if classes[i] == "FAKE" else "Real"
print(f"{label} ({probs[i]*100:.0f}% confident)")
```

Run it:

```bash
python detect.py my_photo.jpg
```

---

## Step 8: Try it on images the model has never seen

Test 20-30 images yourself: your own phone photos (real) and images from any AI tool (fake). If it does badly here, it is overfitting to CIFAKE's style. That's expected and leads to Step 9.

---

## Step 9: Round 2, make it work on realistic images

CIFAKE is tiny and uniform, so the model learns "CIFAKE style", not "AI style". To improve:

1. **Get bigger, more varied data.** Search Hugging Face / Kaggle for datasets such as **GenImage**, **ArtiFact** or **Community Forensics**. Use a subset (20-50k images) so it fits Colab, and check each license and current availability.
2. **Add your own real photos.** 500-2000 pictures from your phone, placed in a `REAL` folder.
3. **Use the same folder layout** so the code works unchanged:
   ```
   data/train/REAL   data/train/FAKE
   data/test/REAL    data/test/FAKE
   ```
4. **Keep the test set from different generators** than the training set (for example, train on Stable Diffusion images, test on Midjourney images). That measures generalization honestly.
5. Change `DATA_DIR`, re-run Steps 3-7. Start Phase B from your old `full_best.pt` to save time.

---

## Step 10 (optional): C2PA metadata check

C2PA is the free, open standard behind Content Credentials. Some AI tools and cameras embed it. Try the free open-source `c2patool` from the Content Authenticity Initiative on your laptop:

```bash
c2patool my_image.jpg
```

If it prints a manifest, the image has provenance data (which may say it was AI-generated). If it prints nothing, that proves nothing, since metadata is often stripped. Use it as an extra signal, and trust your model for the pixels.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` to 32 or 16, then restart the runtime and re-run cells |
| Session disconnected | Re-run Steps 1-3 cells, then the same `run_training(...)` cell. It resumes |
| Training very slow | Lower `MAX_TRAIN`, make sure the data is in `/content/data` (not Drive) |
| `No such file` for data | Re-run the `unzip` cell, then `!ls /content/data` |
| Accuracy stuck near 50% | Check `CLASSES` printed correctly, and that both REAL and FAKE folders have images |
| Great test accuracy but bad on your own images | Normal for CIFAKE. Do Step 9 |
| ONNX export error | `pip install onnxscript`, or remove `dynamo=False` |

---

## Quick glossary

- **Fine-tuning:** continuing to train an already-trained model on your own task.
- **Epoch:** one full pass over the training images.
- **Checkpoint:** a saved snapshot of the model so training can resume.
- **Overfitting:** doing great on training data but poorly on new data.
- **ONNX:** a portable model file format that runs on CPU with `onnxruntime`.
- **Mixed precision (`autocast`):** uses 16-bit math on the GPU to train about 2x faster.

---

*Limits to remember: no detector is 100% reliable, and results shift as new AI generators appear. Treat the output as a probability, not proof.*