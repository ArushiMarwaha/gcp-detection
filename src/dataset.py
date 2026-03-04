import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

shape_to_id = {
    "Cross": 0,
    "Square": 1,
    "L-Shaped": 2
}

class GCPDataset(Dataset):

    def __init__(self, data_list, train_path, img_size=256):

        self.data = data_list
        self.train_path = train_path
        self.img_size = img_size

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        key, value = self.data[idx]

        img_path = os.path.join(self.train_path, key)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        # Extract coordinates
        x = value["mark"]["x"]
        y = value["mark"]["y"]

        # Normalize coordinates
        x = x / w
        y = y / h

        # Shape label
        shape = shape_to_id[value["verified_shape"]]

        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize pixel values
        img = img / 255.0

        # ImageNet normalization
        img = (img - self.mean) / self.std

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        coords = torch.tensor([x, y], dtype=torch.float32)
        shape = torch.tensor(shape, dtype=torch.long)

        return img, coords, shape