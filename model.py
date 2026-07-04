import torch
import torch.nn as nn

class CloudNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.encoder(x)