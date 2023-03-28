import torch
from torch import nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self,opt,image_shape):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

