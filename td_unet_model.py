import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1),
                 norm_layer=nn.BatchNorm3d, activation=F.elu):
        super(ConvBlock, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.norm_layer = norm_layer

        self.conv_a = nn.Conv3d(self.input_channel, self.output_channel, self.kernel_size, stride=stride,
                                padding=padding)
        self.bn_b = norm_layer(self.output_channel)
        self.conv_b = nn.Conv3d(self.output_channel, self.output_channel, self.kernel_size, stride=stride,
                                padding=padding)
        self.bn_a = norm_layer(self.output_channel)

    def forward(self, inputs):
        x = self.conv_a(inputs)
        x = self.bn_a(x)
        x = self.activation(x)
        x = self.conv_b(x)
        x = self.bn_b(x)
        x = self.activation(x)

        return x


# 3D Encoder
class UNet(nn.Module):
    def __init__(self, input_channel, output_channel,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1, activation=F.elu):
        super(UNet, self).__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation
        self.norm_layer = nn.BatchNorm3d
        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), return_indices=True)
        self.unpool = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)

        # TODO change kernel size to figure out long-term and short-term effect
        # Down
        # 1
        self.l1 = ConvBlock(self.input_channel, 64)
        # 2
        self.l2 = ConvBlock(64, 128)
        # 3
        self.l3 = ConvBlock(128, 256)
        # 4
        self.l4 = ConvBlock(256, 512)
        # 5
        self.l5 = ConvBlock(512, 1024)

        # up
        # 4
        self.up1 = nn.Conv3d(1024, 512, (1, 1, 1),
                             stride=(1, 1, 1), padding=0, groups=self.groups)
        self.l6 = ConvBlock(1024, 512)
        # 3
        self.up2 = nn.Conv3d(512, 256, (1, 1, 1),
                             stride=(1, 1, 1), padding=0, groups=self.groups)
        self.l7 = ConvBlock(512, 256)
        # 2
        self.up3 = nn.Conv3d(256, 128, (1, 1, 1),
                             stride=(1, 1, 1), padding=0, groups=self.groups)
        self.l8 = ConvBlock(256, 128)
        # 1
        self.up4 = nn.Conv3d(128, 64, (1, 1, 1),
                             stride=(1, 1, 1), padding=0, groups=self.groups)
        self.l9 = ConvBlock(128, 64)
        self.l10 = nn.Conv3d(64, self.output_channel, self.kernel_size,
                             stride=(1, 1, 1), padding=self.padding, groups=self.groups)

    def forward(self, inputs):

        x = inputs
        conv_out = []  # save conv result

        # down
        for down in [self.l1, self.l2, self.l3, self.l4]:
            x = down(x)
            conv_out.append(x)
            x, idx = self.pool(x)

        conv_out.reverse()

        # up
        ups = [self.up1, self.up2, self.up3, self.up4]
        for i, up in enumerate([self.l5, self.l6, self.l7, self.l8]):
            x = up(x)
            x = ups[i](x)
            x = self.unpool(x)
            x = torch.cat((conv_out[i], x), 1)

        x = self.l9(x)
        x = self.l10(x)

        return x


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# TODO Loss
class ULoss(nn.Module):
    def __init__(self):
        super(ULoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)
