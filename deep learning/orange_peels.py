"""
File: orange_peels.py

This file contains the dependencies for the age classification of dried orange peels.
- Constants: a class containing all the constants used in the project
- OrangePeelsDataset: a custom dataset class for the dried orange peels dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
import numpy as np
import pandas as pd
import random
import os

class Constants:
    # image constants
    IMG_DIR="./data_orange_peels"
    # IMG_SIZE=(3024,4032)
    IMG_SIZE=(504,378)
    # IMG_MAG=0.125
    IMG_MAG=1
    IMG_CON=0.1 # image contrast

    # model constants
    RESNET_MEAN=np.array([0.485, 0.456, 0.406])
    RESNET_STD=np.array([0.229, 0.224, 0.225])
    BATCH_SIZE=32
    EPOCHS=20
    LR=0.001

    # class encoder and decoder constants
    ENCODER = {6 : 0, 10 : 1, 15 : 2, 20 : 3}
    DECODER = {v:k for k, v in ENCODER.items()}

class OrangePeelsDataset(Dataset):
    def __init__(self, img_dir, class_size=None, transform=None, target_transform=None, stats=False):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # store paths and labels in DataFrame
        total = 0
        data = { "image_path" : [], "label" : [] }
        for subdir in os.listdir(img_dir):
            files = [
                f for f in os.listdir(os.path.join(img_dir,subdir)) if f.endswith(".JPG") or f.endswith(".png")]
            random.shuffle(files)
            total += len(files)

            added = 0
            for idx, file in enumerate(files):
                if class_size is not None:
                    if idx >= class_size:
                        break

                data["image_path"].append(os.path.join(img_dir, subdir, file))
                data["label"].append(int(subdir))
                added += 1

            if stats:
                print(f"Class: {subdir}\tAvailable: {len(files)}\t\tAdded: {added}")

        self.annotations = pd.DataFrame(data).sample(frac = 1).reset_index(drop = True)
        if stats:
            print(f"\t\tTotal: {total}\t\tTotal: {len(self.annotations)}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = read_image(self.annotations.iloc[idx, 0]).float()
        label = Constants.ENCODER[self.annotations.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label