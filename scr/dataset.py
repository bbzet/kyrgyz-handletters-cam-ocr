import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class CustomKyrgyzDataset(Dataset):
    def __init__(self, csv_path, train=True, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = row.iloc[1:].values.astype(np.uint8).reshape(50, 50)
        img = Image.fromarray(pixels)

        if self.transform:
            img = self.transform(img)

        if self.train:
            label = int(row.iloc[0]) - 1  # метки от 1 до 36 → [0–35]
            return img, label
        else:
            return img
