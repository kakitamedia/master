import torch
import torch.nn as nn

from .base_networks import *
from model.utils.misc import _sigmoid

# class Discriminator(nn.Module):
#     def __init__(self, cfg):
#         super(Discriminator, self).__init__()

#         norm = cfg.MODEL.DISCRIMINATOR_NORM
#         activation = cfg.MODEL.DISCRIMINAOTR_ACTIVATION
#         output_channel = len(cfg.MODEL.DOWN_RATIOS)
#         if cfg.MODEL.DETECTOR_TYPE == 'hourglass':
#             input_channels = 256
#         else:
#             input_channels = 64

#         self.conv_blocks = nn.Sequential(
#             ConvBlock(input_channels, input_channels, 3, 1, 1, activation=activation, norm=norm),
#             ConvBlock(input_channels, input_channels, 3, 1, 1, activation=activation, norm=norm),
#             ConvBlock(input_channels, input_channels, 3, 1, 1, activation=activation, norm=norm),
#             ConvBlock(input_channels, input_channels*2, 3, 1, 1, activation=activation, norm=norm),
#             ConvBlock(input_channels*2, input_channels*2, 3, 1, 1, activation=activation, norm=norm),
#             ConvBlock(input_channels*2, input_channels*2, 3, 1, 1, activation=activation, norm=norm),
#         )
#         self.out_conv = ConvBlock(input_channels*2, output_channel, 3, 1, 1, activation=None, norm=None)


#     def forward(self, x):
#         x = self.conv_blocks(x)
#         x = self.out_conv(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self, cfg, num_layers):
        super(Discriminator, self).__init__()

        feat = cfg.MODEL.EXTRACTOR_FEAT
        input_channel = cfg.MODEL.DISCRIMINATOR_INPUT_CHANNEL

        self.conv_blocks = []
        self.conv_blocks.append(ResidualBlock(input_channel, feat, kernel_size=3, stride=1, padding=1, activation=cfg.MODEL.DISCRIMINATOR_ACTIVATION, norm=cfg.MODEL.DISCRIMINATOR_NORM))
        for _ in range(num_layers-1):
            self.conv_blocks.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=cfg.MODEL.DISCRIMINATOR_ACTIVATION, norm=cfg.MODEL.DISCRIMINATOR_NORM))
        
        self.conv_blocks.append(ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=None, norm=None))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.input_channel = input_channel

    def forward(self, x):
        return self.conv_blocks(x[:, :self.input_channel, :, :])


class UNetDiscriminator(nn.Module):
    def __init__(self, cfg, input_channels=64):
        super(UNetDiscriminator, self).__init__()

        norm = cfg.MODEL.DISCRIMINATOR_NORM
        activation = cfg.MODEL.DISCRIMINAOTR_ACTIVATION
        output_channel = len(cfg.MODEL.DOWN_RATIOS)

        self.conv_blocks = nn.ModuleList([
            ConvBlock(input_channels*1, input_channels*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(input_channels*1, input_channels*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(input_channels*1, input_channels*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(input_channels*1, input_channels*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(input_channels*1, input_channels*1, 3, 1, 1, activation=activation, norm=norm),

        ])

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(input_channels*1, input_channels*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(input_channels*2, input_channels*1, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(input_channels*1, input_channels*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(input_channels*2, input_channels*1, 3, 1, 1, activation=activation, norm=norm),
        ])

        self.out_conv = nn.ModuleList([
            ConvBlock(input_channels*1, output_channel, 1, 1, 0, activation=None, norm=None),
            ConvBlock(input_channels*1, output_channel, 1, 1, 0, activation=None, norm=None),
            ConvBlock(input_channels*1, output_channel, 1, 1, 0, activation=None, norm=None),
        ])

    def forward(self, x):
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        predictions = []
        predictions.append(self.out_conv[0](x))
        j = 1
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
            else:
                predictions.append(self.out_conv[j](x))
                j += 1
        
        return predictions


class BeganDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(BeganDiscriminator, self).__init__()

        self.input_channels = 64
        self.middle_channels = 128
        self.latent_size = 64

        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 1, 1, 0, activation='elu'),
            nn.AvgPool2d(2, 2),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels*2, 1, 1, 0, activation='elu'),
            nn.AvgPool2d(2, 2),
            ConvBlock(self.middle_channels*2, self.middle_channels*2, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels*2, self.middle_channels*2, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels*2, self.middle_channels*3, 1, 1, 0, activation='elu'),
            nn.AvgPool2d(2, 2),
            ConvBlock(self.middle_channels*3, self.middle_channels*3, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels*3, self.middle_channels*3, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels*3, self.middle_channels*4, 1, 1, 0, activation='elu'),
            nn.AvgPool2d(2, 2),
            ConvBlock(self.middle_channels*4, self.middle_channels*4, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels*4, self.middle_channels*4, 3, 1, 1, activation='elu'),
        )

        self.fc = nn.Sequential(
            nn.Linear(8*8*4*self.middle_channels, self.latent_size),
            nn.Linear(self.latent_size, 8*8*self.middle_channels),
        )

        self.decoder = nn.Sequential(
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            nn.Upsample(scale_factor=2),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            nn.Upsample(scale_factor=2),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            nn.Upsample(scale_factor=2),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            nn.Upsample(scale_factor=2),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.middle_channels, 3, 1, 1, activation='elu'),
            ConvBlock(self.middle_channels, self.input_channels, 3, 1, 1, activation=None),
        )

    def forward(self, x):
        x = self.encoder(x)
        shape = x.shape
        x = x.view(shape[0], -1)
        x = self.fc(x)
        x = x.view(shape[0], -1, shape[2], shape[3])
        x = self.decoder(x)

        return x
