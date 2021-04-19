import torch.nn as nn

from .base_networks import *


params = {
    1: [3, 1, 1],
    2: [6, 2, 2],
    4: [8, 4, 2],
    8: [12, 8, 4]
}

params_rfmatch = {
    1: [7, 1, 3],
    2: [13, 2, 6],
    4: [25, 4, 12],
}

params_rfmatch_dilation = {
    1: [7, 1, 3, 1],
    2: [7, 2, 6, 2],
    4: [7, 4, 12, 4],
}

class Extractor(nn.Module):
    def __init__(self, cfg, num_layers, down_ratio):
        super(Extractor, self).__init__()

        feat = cfg.MODEL.EXTRACTOR_FEAT
        activation = cfg.MODEL.DETECTOR_ACTIVATION
        norm = cfg.MODEL.DETECTOR_NORM
        output_channel = cfg.MODEL.EXTRACTOR_OUTPUT_CHANNEL

        # kernel, stride, padding = params[down_ratio]
 
        self.layers = [ConvBlock(3, feat, kernel_size=7, stride=down_ratio, padding=3, bias=False, activation=activation, norm=norm)]
        # self.layers.append(ResidualBlock(feat, feat, kernel_size=kernel, stride=stride, padding=padding, activation=activation, norm=norm))

        for _ in range(num_layers):
            self.layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        self.layers.append(ConvBlock(feat, output_channel, kernel_size=3, stride=1, padding=1, bias=False, activation=activation, norm=norm))

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.layers(x)


class Extractor_RFMatch(nn.Module):
    def __init__(self, cfg, num_layers, down_ratio):
        super(Extractor_RFMatch, self).__init__()
        
        feat = cfg.MODEL.EXTRACTOR_FEAT
        activation = cfg.MODEL.DETECTOR_ACTIVATION
        norm = cfg.MODEL.DETECTOR_NORM

        kernel, stride, padding = params_rfmatch[down_ratio]

        self.layers = [ConvBlock(3, feat, kernel_size=kernel, stride=stride, padding=padding, activation=activation, norm=norm)]
        self.layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        for _ in range(num_layers):
            self.layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))
        
        self.layers.append(ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Extractor_RFMatch_Dilation(nn.Module):
    def __init__(self, cfg, num_layers, down_ratio):
        super(Extractor_RFMatch_Dilation, self).__init__()

        feat = cfg.MODEL.EXTRACTOR_FEAT
        activation = cfg.MODEL.DETECTOR_ACTIVATION
        norm = cfg.MODEL.DETECTOR_NORM

        kernel, stride, padding, dilation = params_rfmatch_dilation[down_ratio]

        self.layers = [ConvBlock(3, feat, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, activation=activation, norm=norm)]
        self.layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        for _ in range(num_layers):
            self.layers.append(ResidualBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))
        
        self.layers.append(ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class SFTExtractor(nn.Module):
    def __init__(self, cfg, num_layers, down_ratio):
        super(SFTExtractor, self).__init__()

        feat = cfg.MODEL.EXTRACTOR_FEAT
        activation = cfg.MODEL.DETECTOR_ACTIVATION
        norm = cfg.MODEL.DETECTOR_NORM

        kernel, stride, padding = params[down_ratio]

        self.layers = [ConvBlock(3, feat, kernel_size=7, stride=down_ratio, padding=3, bias=False, activation=activation, norm=norm)]
        # self.layers.append(ResidualBlock(feat, feat, kernel_size=kernel, stride=stride, padding=padding, activation=activation, norm=norm))

        for _ in range(num_layers):
            self.layers.append(SFTBlock(feat, feat, kernel_size=3, stride=1, padding=1, activation=activation, norm=norm))

        self.layers.append(ConvBlock(feat, feat, kernel_size=3, stride=1, padding=1, bias=False, activation=activation, norm=norm))

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.layers(x)


class UNetExtractor(nn.Module):
    def __init__(self, down_ratio, feat=64):
        super(UNetExtractor, self).__init__()

        self.down_ratio = down_ratio

        self.layers = [ConvBlock(3, feat, kernel_size=7, stride=1, padding=3, bias=False, activation='relu', norm='batch')]
        self.layers.append(UNetBlock(feat, activation, norm))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        outputs = self.layers(x)


class UNetBlock(nn.Module):
    def __init__(self, feat, activation, norm):
        super(UNetBlock, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, norm=norm),

        ])

        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(feat*1, feat*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(feat*1, feat*1, 3, 1, 1, activation=activation, norm=norm),
        ])

        self.out_conv = nn.ModuleList([
            ConvBlock(feat*1, feat, 1, 1, 0, activation=None, norm=None),
            ConvBlock(feat*1, feat, 1, 1, 0, activation=None, norm=None),
            ConvBlock(feat*1, feat, 1, 1, 0, activation=None, norm=None),
        ])

    def forward(self, x):
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        outputs = []
        outputs.append(self.out_conv[0](x))
        j = 1
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
            else:
                outputs.append(self.out_conv[j](x))
                j += 1
        
        return outputs
    


class ResnetBlock(nn.Module):
    def __init__(self, channel, num_layers, down=False):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(ConvBlock(channel, channel, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm='batch'))
        if down:
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)