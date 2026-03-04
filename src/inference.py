import os
import json
import torch
import cv2

from model import GCPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GCPModel().to(device)
model.load_state_dict(torch.load("gcp_model.pth", map_location=device))

model.eval()

id_to_shape = {
0:"Cross",
1:"Square",
2:"L-Shaped"
}

test_path = "C:/Users/arush/Desktop/Data/test_dataset"

test_images = []

for root,dirs,files in os.walk(test_path):

    for file in files:

        if file.endswith(".JPG") or file.endswith(".jpg"):

            full_path = os.path.join(root,file)
            rel_path = os.path.relpath(full_path,test_path).replace("\\","/")

            test_images.append((full_path,rel_path))

predictions = {}

for full_path,rel_path in test_images:

    img = cv2.imread(full_path)

    h,w,_ = img.shape

    img = cv2.resize(img,(256,256))
    img = img/255.0

    img = torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():

        pred_coords,pred_shape = model(img)

    pred_coords = pred_coords.cpu().numpy()[0]

    x = float(pred_coords[0]*w)
    y = float(pred_coords[1]*h)

    shape = pred_shape.argmax(dim=1).item()

    predictions[rel_path] = {
        "mark":{"x":x,"y":y},
        "verified_shape":id_to_shape[shape]
    }

with open("../outputs/predictions.json","w") as f:
    json.dump(predictions,f,indent=4)

print("predictions.json saved")