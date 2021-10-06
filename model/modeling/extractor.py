import torch.nn as nn

from .base_network import *


def build_extractors(cfg):
    model_type = cfg.MODEL.EXTRACTOR.TYPE
    if model_type == 'unet':
        return nn.ModuleList([UNetExtractor(cfg, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS])

    else:
        raise NotImplementedError('cfg.MODEL.EXTRACTOR.TYPE: {} is not supported'.format(model_type))


class UNetExtractor(nn.Module):
    def __init__(self, cfg, down_ratio):
        super(UNetExtractor, self).__init__()

        feat = cfg.MODEL.DETECTOR.INPUT_CHANNEL
        activation = cfg.MODEL.DETECTOR.ACTIVATION
        normalization = cfg.MODEL.DETECTOR.NORMALIZATION

        self.layers = [ConvBlock(3, feat, kernel_size=7, stride=down_ratio, padding=3, bias=False, activation='relu', normalization=normalization)]
        self.layers.append(UNetBlock(feat, activation=activation, normalization=normalization))
        self.layers.append(ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, bias=False, activation=activation, normalization=normalization))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class UNetBlock(nn.Module):
    def __init__(self, feat, activation='relu', normalization='batch'):
        super(UNetBlock, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, normalization=normalization),
            ConvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, normalization=normalization),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, normalization=normalization),
            ConvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, normalization=normalization),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, normalization=normalization),

        ])

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, normalization=normalization),
            ConvBlock(feat*2, feat*1, 3, 1, 1, activation=activation, normalization=normalization),
            DeconvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, normalization=normalization),
            ConvBlock(feat*2, feat*1, 3, 1, 1, activation=activation, normalization=normalization),
        ])


    def forward(self, x):
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
        
        return x