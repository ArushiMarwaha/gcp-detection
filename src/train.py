import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GCPDataset
from model import GCPModel

train_path = "../train_dataset"
json_path = os.path.join(train_path,"curated_gcp_marks.json")

with open(json_path) as f:
    labels = json.load(f)

valid_data = []

for key,value in labels.items():

    img_path = os.path.join(train_path,key)

    if os.path.exists(img_path):

        if "verified_shape" in value and "mark" in value:
            valid_data.append((key,value))

dataset = GCPDataset(valid_data,train_path)

loader = DataLoader(dataset,batch_size=8,shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GCPModel().to(device)

coord_loss = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs = 3

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for imgs,coords,shapes in loader:

        imgs = imgs.to(device)
        coords = coords.to(device)
        shapes = shapes.to(device)

        pred_coords,pred_shapes = model(imgs)

        loss1 = coord_loss(pred_coords,coords)
        loss2 = class_loss(pred_shapes,shapes)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss/len(loader))

torch.save(model.state_dict(),"../outputs/gcp_model.pth")