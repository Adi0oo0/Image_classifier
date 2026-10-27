# all images are of size 32x32x3 so for now
# no resize required
# data selection for now is random, but later we will implement a more robust data selection method
# label is like 0 -> 'ai', 1 -> 'real'


import numpy as np
import torch
import os
import random
from PIL import Image

img = Image.open('data/test/ai/0 (2).jpg')

img_arr = torch.tensor(np.array(img))    # 32x32x3

def get_imgs_tensor(split: str = "train", bin: str = 'ai', quantity: int = 10):
    try:
        path = f"data/{split}/{bin.lower()}/"
    except FileNotFoundError:
        print(f"Folder {path} not found")
        return None

    ran_imgs = random.sample(os.listdir(path), quantity)
    print(f"Selected {quantity} images from {path}")
    return [torch.tensor(np.array(Image.open(path + img))) for img in ran_imgs]


