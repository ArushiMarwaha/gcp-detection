import torch.nn as nn
import torchvision.models as models

class GCPModel(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.fc_coords = nn.Linear(512,2)
        self.fc_shape = nn.Linear(512,3)

    def forward(self,x):

        x = self.features(x)
        x = x.view(x.size(0),-1)

        coords = self.fc_coords(x)
        shape = self.fc_shape(x)

        return coords, shape