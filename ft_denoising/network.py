import torch
from typing import List
import numpy as np

class SimpleFTDenoising(torch.nn.Module):
    def __init__(
        self,
        layers: int,
        in_channel: List[int],
        out_channel: List[int],
    ):
        super(SimpleFTDenoising, self).__init__()

        if len(in_channel) != len(out_channel) or len(in_channel) != layers:
            raise ValueError("The input parameters must have the same length")

        # the network is composed by a series of fully connected layers
        self.fc_layers = torch.nn.ModuleList()

        # add convolution
        self.fc_layers.append(torch.nn.Flatten())
        for i in range(layers):
            self.fc_layers.append(torch.nn.Linear(in_channel[i], out_channel[i]))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(0.05))

        # Output layer
        self.fc_layers.append(torch.nn.Linear(out_channel[-1], 6))
        self.name = "SimpleFTDenoising"

    def forward(self, x):
        # Apply fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x
