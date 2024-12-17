import torch.nn.utils.spectral_norm as spectral_norm
from torch import nn
from utils.FixedConv import UpSampling2d, DownSampling2d
from utils.WeightNormalization import WeightNormalization

class ConvolutionalBlock(nn.Module):
    """
        Convolutional blocks for upsampling block and downsampling block 
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 is_downsampling: bool = True,
                 add_activation: bool = True,
                 apply_dropout=False,
                 Keep = False,
                 bias = True,
                 **kwarg):
        super(ConvolutionalBlock, self).__init__()
        layers = list()

        # For downsampling blocks and using fixed smooth convolution
        if is_downsampling and not Keep:

            layers.append(DownSampling2d(in_channel, out_channel, **kwarg, dilation=1, groups=1,
                          bias=True, padding_mode="reflect",
                          order=0, hold_mode='hold_last', bias_mode='bias_first'))
            layers.append(nn.InstanceNorm2d(out_channel))
            layers.append(
                WeightNormalization(
                    out_channel,
                )
            )
            layers.append(
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                if add_activation
                else nn.Identity()
            )
        # For downsampling but only convolution 
        elif is_downsampling and Keep:

            layers.append(nn.Conv2d(
                in_channel, out_channel, padding_mode="reflect",  **kwarg))
            layers.append(nn.InstanceNorm2d(out_channel))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True) if add_activation else nn.Identity())
        # For upsampling and using fixed smooth convolution
        elif not is_downsampling and not Keep:

            layers.append(UpSampling2d(in_channel, out_channel, **kwarg , dilation=1, groups=1,
                           bias=True, padding_mode='zeros',
                           order=0, hold_mode='hold_first', bias_mode='bias_first'))

            layers.append(nn.InstanceNorm2d(out_channel))

            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True) if add_activation else nn.Identity())
        # For Upsampling and but only transposed convolution 
        else:
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, **kwarg))
            layers.append(nn.InstanceNorm2d(out_channel))

            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True) if add_activation else nn.Identity())

        self.ConvB = nn.Sequential(*layers)
    def forward(self, x):
        x = self.ConvB(x)
        return x
    
class ResidualBlock(nn.Module):
    """
        Residual block that learn the residual mapping between
        the input and output of the block 
    """
    def __init__(self,
                 channel: int,
                 **kwarg):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channel, channel, add_activation=True, Keep =True,apply_dropout=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channel, channel, add_activation=False, Keep =True, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
        Generator 
        Consists of 2 layers of downsampling layer, 
        6 residual blocks and then 2 upsampling layer. 
    """
    
    def __init__(self, 
                 input_channel: int, 
                 features: int = 64, 
                 num_residuals: int = 6):
        super().__init__()
        self.first_layer = nn.Sequential(
            DownSampling2d(input_channel, features, kernel_size=7, stride=1, padding=3, dilation=1, groups=1,
                          bias=True, padding_mode="reflect",
                          order=0, hold_mode='hold_last', bias_mode='bias_first'),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.downsampling_layer = nn.Sequential(
            ConvolutionalBlock(features, features*2, is_downsampling = True, add_activation= True, apply_dropout=True, kernel_size=3, stride=2, padding=1),
            ConvolutionalBlock(features*2, features*4, is_downsampling = True, add_activation= True, Keep =True, apply_dropout=True, kernel_size=3, stride=2, padding=1)
        )
        self.residual_layer = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residuals)]
        )
        self.upsampling_layer = nn.Sequential(
            ConvolutionalBlock(features*4, features*2, is_downsampling = False, add_activation= True, Keep = True,kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvolutionalBlock(features*2, features, is_downsampling = False, add_activation= True, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(features, input_channel, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )
    
    def forward(self, x):

        x = self.first_layer(x)
        x = self.downsampling_layer(x)
        x = self.residual_layer(x)
        x = self.upsampling_layer(x)
        x = self.last_layer(x)
        return x

class Discriminator(nn.Module):
    """
        Discriminator 
        Consists of 3 Convolution-InstanceNorm-LeakyReLU block
    """

    def __init__(self, 
                 input_channel: int, 
                 features: int = 64, 
                 num_layers: int = 4):
        super().__init__()

        layers = list()
        layers.append(spectral_norm(nn.Conv2d(input_channel, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect")))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, num_layers):
            in_channel = features * 2 ** (i-1)
            out_channel = in_channel * 2

            if i == num_layers - 1:
                layers.append(self.Convlayer(in_channel, out_channel, 4, 1))
            else:
                layers.append(self.Convlayer(in_channel, out_channel, 4, 2))

        layers.append(nn.Conv2d(out_channel, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())  

        self.Disc = nn.Sequential(*layers)

    def Convlayer(self, in_ch, out_ch, kernel_size, stride, use_leaky=True, use_inst_norm=True, use_pad=True):
        if use_pad:
            conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, 1, bias=True, padding_mode="reflect"))
        else:
            conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=True, padding_mode="reflect"))

        if use_leaky:
            actv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            actv = nn.ReLU(inplace=True)

        if use_inst_norm:
            norm = nn.InstanceNorm2d(out_ch)
        else:
            norm = nn.BatchNorm2d(out_ch)

        return nn.Sequential(conv, norm, actv)
    
    def forward(self, x):
        return self.Disc(x)