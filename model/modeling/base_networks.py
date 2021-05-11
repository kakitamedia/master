import torch
import torch.nn as nn
import math

from mmcv.ops.deform_conv import DeformConv2d


class BlockBase(nn.Module):
    def __init__(self, bias, activation, norm):
        super(BlockBase, self).__init__()

        ### Nomalizing layer
        if self.norm =='batch':
            self.norm = nn.BatchNorm2d(output_dim)
        elif self.norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif self.norm == 'group':
            self.norm = nn.GroupNorm(32, output_dim)
        elif self.norm == 'spectral':
            self.norm = None
            self.layer = nn.utils.spectral_norm(self.layer)
        elif norm == None:
            self.norm = None

        ### Activation layer
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'prelu':
            self.act = nn.PReLU(init=0.01)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.01, True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activaton == None:
            self.act = None

        ### Initialize weights
        if activation == 'relu':
            nn.init.kaiming_normal_(self.layer.weight, nonlinearity='relu')
        elif activaton == 'prelu' or activation == 'lrelu':
            nn.init.kaiming_normal_(self.layer.weight, a=0.01, nonlinearity='leaky_relu')
        elif activation == 'tanh':
            nn.init.xavier_normal_(self.layer.weight, gain=5/3)
        else:
            nn.init.xavier_normal_(self.layer.weight, gain=1)

        if bias:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)      

        return x


class DenseBlock(BlockBase):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu', norm='batch'):
        self.layer = nn.Linear(input_dim, output_dim, bias=bias)
        super(BlockBase, self).__init__(bias, activation, norm)

        ### Overwrite normalizing layer for 1D version
        self.norm = norm
        if self.norm =='batch':
            self.norm = nn.BatchNorm1d(output_dim)
        elif self.norm == 'instance':
            self.norm = nn.InstanceNorm1d(output_dim)


class ConvBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, activation='relu', norm=None):
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation, groups, bias=bias)
        super(BlockBase, self).__init__(bias, activation, norm)


class DeconvBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, activation='relu', norm=None):
        self.layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, dilation, groups, bias=bias)
        super(BlockBase, self).__init__(bias, activation, norm)


class DeformableConvBlock(BlockBase):
    def __init__(input_dim, output_dim, offset_dim=None, kernel_size=3, stride=1, padding=1, deform_groups=1, bias=False, activation='relu', norm=None):
        if offsett_dim is None:
            offset_dim = input_dim
        
        self.layer = DeformConv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias)
        self.offset_conv = nn.Conv2d(offset_dim, deform_groups * 2 * kernel_size**2, kernel_size, stride, padding, bias=True)
            
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

        super(BlockBase, self).__init__(bias, activation, norm)


    def forward(self, x, offset=None):
        if offset is None:
            offset = self.offset_conv(x)
        else:
            offset = self.offset_conv(offset)

        x = self.layer(x, offset)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)      

        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_convs=2, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm=None):
        super(ResidualBlock, self).__init__()

        self.num_convs = num_convs

        self.layers, self.norms, self.acts = [], [], []
        self.layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias))
        for _ in range(num_convs - 1):
            self.layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding, bias=bias))

        for i in range(num_convs):
            ### Normalizing layer
            if norm == 'batch':
                self.norms.append(nn.BatchNorm2d(output_dim))
            elif norm == 'instance':
                self.norms.append(nn.InstansNorm2d(output_dim))
            elif norm == 'group':
                self.norms.append(nn.GroupNorm(32, output_dim))
            elif norm == 'spectral':
                self.norms.append(None)
                self.layers[i] = nn.utils.spectral_norm(self.layers[i])
            elif norm == None:
                self.norms.append(None)
            else:
                raise Exception('norm={} is not implemented.'.format(norm))

            ### Activation layer
            if activation == 'relu':
                self.acts.append(nn.ReLU(True))
            elif activation == 'lrelu':
                self.acts.append(nn.LeakyReLU(0.01, True))
            elif activation == 'prelu':
                self.acts.append(nn.PReLU(init=0.01))
            elif activation == 'tanh':
                self.acts.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.acts.append(nn.Sigmoid())
            elif activaton == None:
                self.acts.append(None)
            else:
                raise Exception('activation={} is not implemented.'.format(activaton))

            ### Initialize weights
            if activation == 'relu':
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            elif activation == 'lrelu' or activation == 'prelu':
                nn.init.kaiming_normal_(self.layers[i].weight, a=0.01, nonlinearity='leaky_relu')
            elif activation == 'tanh':
                nn.init.xavier_normal_(self.layers[i].weight, gain=5/3)
            elif activation == 'sigmoid':
                nn.init.xavier_normal_(self.layers[i].weight, gain=1)
            elif activation == None:
                nn.init.xaview_normal_(self.layers[i].weight, gain=1)
            else:
                raise Exception('activation={} is not implemented.'.format(activaton))

            if bias:
                nn.init.zeros(self.layers[i].bias)


        if input_dim != output_dim or stride != 1:
            self.skip_layer = nn.Conv2d(input_dim, output_dim, 1, stride, 0, bias=bias)
        else:
            self.skip_layer = None

        if bias:
            nn.init.zeros_(self.skip_layer.bias)

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        for i in range(self.num_convs):
            x = self.layers[i](x)

            if i == self.num_convs - 1:
                x = x + residual

            if self.norms[i] is not None:
                x = self.norms[i](x)
            
            if self.acts[i] is not None:
                x = self.acts[i](x)

        return x


class FixUpResidualBlock(ResidualBlock):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm=None, num_residuals=1):
        super(ResidualBlock, self).__init__(self, input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, activation=activation, norm=norm)

        for param in self.conv1.parameters():
            param = param / ( num_residuals ** (1 / 2) )
        self.conv2.weight.data.zero_()

        if bias:
            self.conv1_bias = nn.Parameter(torch.zeros(1))
            self.conv2_bias = nn.Parameter(torch.zeros(1))
            self.act1_bias = nn.Parameter(torch.zeros(1))
            self.act2_bias = nn.Parameter(torch.zeros(1))
        else:
            self.conv1_bias = None
            self.conv2_bias = None
            self.act1_bias = None
            self.act2_bias = None

        self.multiplier = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        if self.conv1_bias is not None:
            x = x + self.conv1_bias

        out = self.layer1(x)

        if self.act1_bias is not None:
            out = out + self.act1_bias

        if self.norm1 is not None:
            out = self.norm1(out)

        if self.act1 is not None:
            out = self.act1(out)

        if self.conv2_bias is not None:
            out = out + self.conv2_bias

        out = self.layer2(out) * self.multiplier

        if self.act2_bias is not None:
            out = out + self.act2_bias

        out = out + residual

        if self.norm2 is not None:
            out = self.norm2(out)

        if self.act2 is not None:
            out = self.act2(out)        

        return out
        


### Backups
class SFTBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm='batch'):
        super(SFTBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = ConvBlock(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, activation=activation, norm=norm)
        self.conv2 = ConvBlock(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=bias, activation=None, norm=norm)

        self.scale_convs = nn.Sequential(
            ConvBlock(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation, norm=norm),
            ConvBlock(output_dim, output_dim, kernel_size=3, stride=1, padding=1, activation=None, norm=norm),
        )

        self.shift_convs = nn.Sequential(
            ConvBlock(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation, norm=norm),
            ConvBlock(output_dim, output_dim, kernel_size=3, stride=1, padding=1, activation=None, norm=norm),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        scale = self.scale_convs(x)
        shift = self.shift_convs(x)

        return x * (1 + scale) + shift


class ResidualUpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False):
        super(ResidualUpBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.ConvTranspose2d(input_dim, output_dim, 4, stride=2, padding=1, bias=bias)
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, stride=1, padding=1, bias=bias)
        self.norm2 = nn.BatchNorm2d(output_dim)

        if input_dim != output_dim or stride != 1:
            self.skip_layer = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, 4, stride=2, padding=1, bias=bias),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.skip_layer = None

        self.act2 = nn.ReLU(True)


    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.act2(x + residual)


class MargeBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm='batch'):
        super(MargeBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm2 = nn.BatchNorm2d(dim)

        if activation == 'relu':
            self.act = nn.ReLU(True)

    def forward(self, a, b):
         a = self.norm1(self.conv1(a))
         b = self.norm2(self.conv2(b))

         return self.act(a+b)



class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=False, activation='relu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=False, activation='relu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=False, activation='relu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=False, activation='relu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0