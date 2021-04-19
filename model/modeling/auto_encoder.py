import torch.nn as nn

from .base_networks import *


class AutoEncoder(nn.Module):
    def __init__(self, input_size=64, feat=64, latent_size=128):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=4, stride=2, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=4, stride=2, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
        )

        self.latent1 = DenseBlock(feat*(input_size//4)**2, latent_size, activation='relu', norm=None)
        
        self.latent2 = DenseBlock(latent_size, feat*(input_size//4)**2, activation='relu', norm=None)

        self.decoder = nn.Sequential(
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
            DeconvBlock(feat, feat, kernel_size=4, stride=2, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
            DeconvBlock(feat, feat, kernel_size=4, stride=2, padding=1, activation='relu', norm=None),
            ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation='relu', norm=None),
        )


    def forward(self, x, inference=False):
        x = self.encoder(x)
        batch, channel, height, width = x.shape
        x = x.view(batch, -1)
        x = self.latent1(x)

        if inference:
            return x

        x = self.latent2(x)
        x = x.view(batch, channel, height, width)
        x = self.decoder(x)

        return x
