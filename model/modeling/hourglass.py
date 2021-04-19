import torch
import torch.nn as nn

from .base_networks import *
from model.utils.misc import _sigmoid
from model.utils.misc import fix_model_state_dict


def make_scalekeep_module(input_dim, output_dim, num_modules):
    layers = [ResidualBlock(input_dim, output_dim, stride=1)]
    for _ in range(1, num_modules):
        layers.append(ResidualBlock(output_dim, output_dim, stride=1))

    return nn.Sequential(*layers)
    

def make_scaledown_module(input_dim, output_dim, num_modules):
    layers = [ResidualBlock(input_dim, output_dim, stride=2)]
    for _ in range(1, num_modules):
        layers.append(ResidualBlock(output_dim, output_dim, stride=1))

    return nn.Sequential(*layers)


def make_scaleup_module(input_dim, output_dim, num_modules):
    layers = []
    for _ in range(num_modules - 1):
        layers.append(ResidualBlock(input_dim, input_dim, stride=1))
    layers.append(ResidualBlock(input_dim, output_dim, stride=1))
    layers.append(nn.Upsample(scale_factor=2))

    return nn.Sequential(*layers)


def make_output_module(input_dim, output_dim, nstack):
    layers = []
    for _ in range(nstack):
        layer = nn.Sequential(
            ConvBlock(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            ConvBlock(input_dim, output_dim, kernel_size=1, stride=1, padding=0, activation=None, norm=None, bias=True),
        )
        layers.append(layer)
    
    return nn.ModuleList(layers)


class Hourglass(nn.Module):
    def __init__(self, n, dims, modules, increase=0):
        super(Hourglass, self).__init__()
        assert n == len(dims) - 1 == len(modules) -1

        self.n = n

        current_num_modules = modules[0]
        next_num_modules = modules[1]

        current_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_scalekeep_module(current_dim, current_dim, current_num_modules)
        self.low1 = make_scaledown_module(current_dim, next_dim, current_num_modules)
        
        if self.n > 1:
            self.low2 = Hourglass(n-1, dims[1:], modules[1:])
        else:
            self.low2 = make_scalekeep_module(next_dim, next_dim, next_num_modules)

        self.up2 = make_scaleup_module(next_dim, current_dim, current_num_modules)


    def forward(self, x):
        residual = self.up1(x)
        x = self.low1(x)
        x = self.low2(x)
        x = self.up2(x)

        return x + residual


class HourglassNet(nn.Module):
    def __init__(self, cfg, increase=0):
        super(Hourglass, self).__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        nstack = cfg.MODEL.NSTACK
        dims = cfg.MODEL.DIMS
        modules = cfg.MODEL.MODULES

        self.init_conv = nn.Sequential(
            ConvBlock(3, 128, kernel_size=7, stride=2, padding=3, norm='batch', activation='relu'),
            ResidualBlock(128, 256, kernel_size=3, stride=2, padding=1),
        )

        self.hourglasses = nn.ModuleList([
            Hourglass(5, dims, modules, increase) for _ in range(nstack)
        ])

        self.prepred_convs = nn.ModuleList([
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, norm='batch', activation='relu')
            for _ in range(nstack)
        ])

        self.merge_conv = nn.ModuleList([
            MargeBlock(256, kernel_size=1, stride=1, padding=0) for _ in range(nstack-1)
        ])

        self.inter_convs = nn.ModuleList([
            ResidualBlock(256, 256) for _ in range(nstack-1)
        ])

        # self.hm_out_conv = make_output_module(256, num_classes, nstack)
        # self.wh_out_conv = make_output_module(256, 2, nstack)
        # self.reg_out_conv = make_output_module(256, 2, nstack)

        self.hm = make_output_module(256, num_classes, nstack)
        self.wh = make_output_module(256, 2, nstack)
        self.reg = make_output_module(256, 2, nstack)

        self.nstack = nstack

        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('Conv2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif classname.find('ConvTranspose2d') != -1:
        #         torch.nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()


    def forward(self, x):
        x = self.init_conv(x)

        outputs = []
        for i in range(self.nstack):
            hourglass = self.hourglasses[i](x)
            pred_features = self.prepred_convs[i](hourglass)

            hm_pred = _sigmoid(self.hm[i](pred_features))
            wh_pred = self.wh[i](pred_features)
            reg_pred = self.reg[i](pred_features)

            outputs.append({'hm': hm_pred, 'wh': wh_pred, 'reg': reg_pred})

            if i < self.nstack - 1:
                x = self.merge_conv[i](x, pred_features)
                x = self.inter_convs[i](x)

        return outputs

class MultiScaleHourglassNet(nn.Module):
    def __init__(self, cfg, increase=0, inference=False):
        super(MultiScaleHourglassNet, self).__init__()

        num_classes = cfg.MODEL.NUM_CLASSES
        nstack = cfg.MODEL.HOURGLASS.NSTACK
        dims = cfg.MODEL.HOURGLASS.DIMS
        modules = cfg.MODEL.HOURGLASS.MODULES

        self.init_convs = nn.ModuleList([
            nn.Sequential(
            ConvBlock(3, 128, kernel_size=7, stride=2, padding=3, norm='batch', activation='relu'),
            ResidualBlock(128, 256, kernel_size=3, stride=2, padding=1),
            )
        ])

        self.hourglasses = nn.ModuleList([
            Hourglass(5, dims, modules, increase) for _ in range(nstack)
        ])

        self.prepred_convs = nn.ModuleList([
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, norm='batch', activation='relu')
            for _ in range(nstack)
        ])

        self.merge_conv = nn.ModuleList([
            MargeBlock(256, kernel_size=1, stride=1, padding=0) for _ in range(nstack-1)
        ])

        self.inter_convs = nn.ModuleList([
            ResidualBlock(256, 256) for _ in range(nstack-1)
        ])

        self.wh_out_conv = make_output_module(256, 2, nstack)
        self.reg_out_conv = make_output_module(256, 2, nstack)
        self.hm_out_conv = make_output_module(256, 80, nstack)

        if not cfg.SOLVER.SCRATCH:
            self.load_state_dict(fix_model_state_dict(torch.load(cfg.PRETRAINED_MODEL, map_location=lambda storage, loc:storage)))
            print('Initialize layers with CenterNet pretrained model'.format(cfg.PRETRAINED_MODEL))

        self.init_convs.extend(nn.ModuleList([
            nn.Sequential(
                ConvBlock(3, 128, kernel_size=7, stride=2, padding=3, norm='batch', activation='relu'),
                ResidualBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                ConvBlock(3, 128, kernel_size=7, stride=1, padding=3, norm='batch', activation='relu'),
                ResidualBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                ConvBlock(3, 128, kernel_size=7, stride=1, padding=3, norm='batch', activation='relu'),
                ResidualUpBlock(128, 256, kernel_size=3, stride=2, padding=2),
            )
        ]))

        if cfg.MODEL.NUM_CLASSES != 80:
            self.hm_out_conv = make_output_module(256, num_classes, nstack)

        self.nstack = nstack

        self.inference = inference


    def forward(self, x, j, d_train=False):
        x = self.init_convs[j](x)
        if d_train:
            return x

        if not self.inference:
            features = x

        outputs = []
        for i in range(self.nstack):
            hourglass = self.hourglasses[i](x)
            pred_features = self.prepred_convs[i](hourglass)

            hm_pred = _sigmoid(self.hm[i](pred_features))
            wh_pred = self.wh[i](pred_features)
            reg_pred = self.reg[i](pred_features)

            outputs.append({'hm': hm_pred, 'wh': wh_pred, 'reg': reg_pred})

            if i < self.nstack - 1:
                x = self.merge_conv[i](x, pred_features)
                x = self.inter_convs[i](x)

        if self.inference:
            return outputs[self.nstack - 1]
        else:
            return outputs, features