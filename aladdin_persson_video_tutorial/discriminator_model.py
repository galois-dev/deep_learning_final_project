import torch
import torch.nn as nn


# A part of the discriminator
class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=4, padding=1, bias=True, padding_mode='reflect'),    # padding mode 'reflect' helps reduce artefacts
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


# Actual discriminator
class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):    # in_channels=3 -> RGB as default; features -> we send an in_channel
        super().__init__()

        # Use conv block for all features
        ## An exception to using the rule of instance norm in every block -> no instance norm
        ### TO DO: can improve by having an arg in self.conv to have instance norm or not
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        # Create layers with blocks 
        layers = []
        in_channels = features[0]   # as we first run it through the initial block, it changes to 64
        for feature in features[1:]:    # skip first feature since that is self.initial
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))    # WHY? last feature has a stride of 1
            in_channels = feature
        
        # Additional conv layer to map between 0 or 1 to indicate real or fake image
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

        # Unwrap layers and pass through a sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x)) # sigmoid to make sure output is between 0 and 1
    

# Test with random data
def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)  # should be a 30 x 30 output ([5, 1, 30, 30]) => each value in the grid sees a 70 x 70 patch in the og image (PatchGAN)


if __name__=='__main__':
    test()
