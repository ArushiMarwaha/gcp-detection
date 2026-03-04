import torch.nn as nn
import torchvision.models as models


class GCPModel(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        # Remove final classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Coordinate regression head
        self.coord_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # Shape classification head
        self.shape_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )


    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        coords = self.coord_head(x)

        shape = self.shape_head(x)

        return coords, shape