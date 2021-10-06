import torch.nn as nn

from model.modeling.base_network import *


def build_discriminator(cfg):
    model_type = cfg.MODEL.DISCRIMINATOR.TYPE
    if model_type == 'residual':
        return ResidualDiscriminator(cfg)
    else:
        raise NotImplementedError('cfg.MODEL.DISCRIMINATOR.TYPE: {} is not supported'.format(model_type))


class ResidualDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(ResidualDiscriminator, self).__init__()

        feat = cfg.MODEL.DISCRIMINATOR.HIDDEN_CHANNEL
        activation = cfg.MODEL.DISCRIMINATOR.ACTIVATION
        normalization = cfg.MODEL.DISCRIMINATOR.NORMALIZATION
        input_channel = cfg.MODEL.DISCRIMINATOR.INPUT_CHANNEL
        output_channel = len(cfg.MODEL.DOWN_RATIOS)

        num_layers = 5

        self.conv_blocks = [ConvBlock(input_channel, feat, kernel_size=7, stride=1, padding=3, bias=True, activation=activation, normalization=normalization)]
        for _ in range(num_layers):
            self.conv_blocks.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization))
        self.conv_blocks.append(ConvBlock(feat, output_channel, kernel_size=3, stride=1, padding=1, activation=None, normalization=None))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.input_channel = input_channel

    def forward(self, x):
        return self.conv_blocks(x[:, :self.input_channel, :, :])
