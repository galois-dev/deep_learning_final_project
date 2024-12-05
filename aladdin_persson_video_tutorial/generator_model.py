import torch
import torch.nn as nn


# A part of the generator that can do downsampling or upsampling
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):   # down = True -> downsampling, down = False -> upsampling: use_act -> use an activation
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)  # keyword args (kwargs) => kernel_size, stride, padding
            if down # use above conv if downsampling, else..
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )
    
    def forward(self, x):
        return self.conv(x)


# Create residual block
class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, stride=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1, stride=1),    # not really sure why act only used in first block, was just the way it is in og paper
        )
    
    def forward(self, x):
        return x + self.block(x)    # possible as we are not changing channels and have same kwargs


# Generator
class Generator(nn.Module):

    def __init__(self, img_channels, num_features=64, num_residuals=9):    # num_residuals=9 if >=256 else 6   
        super().__init__()

        # Create block that does not use InstanceNorm
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )

        # Downsample
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*2*2, down=True, kernel_size=3, stride=2, padding=1),
            ]
        )

        # Residual blocks -> do not change size/shape
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)],  # unwrappping the 9 residual blocks
        )

        # Upsample
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1), # output_padding adds another padding after ConvBlock
                ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1), 
            ]
        )

        # Convert to RGB
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x)) # tanh -> [-1, 1]


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape) # should be exactly identical as input => 256 x 256


if __name__ == '__main__':
    test()