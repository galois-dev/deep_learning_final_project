import torch
import torch.nn as nn


class WeightNormalization(nn.Module):
    def __init__(self, in_channels, epsilon=1e-6):
        super(WeightNormalization, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_channels))  
        self.scaling = nn.Parameter(torch.ones(in_channels)) 
        self.epsilon = epsilon

    def forward(self, x):
        unsqueezed = False
        if x.dim() == 3: 
            x = x.unsqueeze(0) 
            unsqueezed = True
        norm = torch.sqrt(torch.sum(x**2, dim=(2, 3), keepdim=True) + self.epsilon)
        scaled = x / norm
        scaled = scaled * self.scaling.view(
            1, -1, 1, 1
        ) 
        if unsqueezed:  
            scaled = scaled.squeeze(0)

        return scaled