import os
import cv2
import torch
from torch.utils.data import Dataset

shape_to_id = {
    "Cross":0,
    "Square":1,
    "L-Shaped":2
}

class GCPDataset(Dataset):

    def __init__(self, data_list, train_path):
        self.data = data_list
        self.train_path = train_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        key, value = self.data[idx]

        img_path = os.path.join(self.train_path, key)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        x = value["mark"]["x"] / w
        y = value["mark"]["y"] / h

        shape = shape_to_id[value["verified_shape"]]

        img = cv2.resize(img,(256,256))
        img = img/255.0

        img = torch.tensor(img,dtype=torch.float32).permute(2,0,1)

        coords = torch.tensor([x,y],dtype=torch.float32)
        shape = torch.tensor(shape)

        return img, coords, shape