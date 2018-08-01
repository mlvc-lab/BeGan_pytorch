import torch.nn as nn
import torch.nn.functional as F


# 3D Encoder
class TdEncoder(nn.Module):
    def __init__(self, depth, input_channel, hidden_size,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1, activation=F.elu):
        super(TdEncoder, self).__init__()

        self.depth = depth
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation

        # TODO change kernel size to figure out long-term and short-term effect
        # 1
        self.l1 = nn.Conv3d(self.input_channel, 64, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 2
        self.l2 = nn.Conv3d(64, 128, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        # 3
        self.l3 = nn.Conv3d(128, 256, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l4 = nn.Conv3d(256, 256, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        # 4
        self.l5 = nn.Conv3d(256, 512, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l6 = nn.Conv3d(512, 512, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.pool4 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # 5
        self.l7 = nn.Conv3d(512, 512, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l8 = nn.Conv3d(512, 512, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.pool5 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.l9 = nn.Linear(self.depth * 512 * 8 * 8, self.hidden_size)

    def forward(self, inputs):
        x = self.activation(self.l1(inputs))
        x = self.activation(self.pool1(x))

        x = self.activation(self.l2(x))
        x = self.activation(self.pool2(x))

        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.pool3(x))

        x = self.activation(self.l5(x))
        x = self.activation(self.l6(x))
        x = self.activation(self.pool4(x))

        x = self.activation(self.l7(x))
        x = self.activation(self.l8(x))
        x = self.activation(self.pool5(x))
        x = x.view(-1, self.depth * 512 * 8 * 8)
        x = self.activation(self.l9(x))

        return x


# 3D Decoder
class TdDecoder(nn.Module):
    def __init__(self, depth, input_channel, hidden_size,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1, activation=F.elu):
        super(TdDecoder, self).__init__()

        self.depth = depth
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation

        # TODO change kernel size to figure out long-term and short-term effect
        # 1
        self.l1 = nn.Linear(self.hidden_size, (512 * self.depth * 8 * 8))
        self.up0 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.l2 = nn.Conv3d(512, 512, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

        # 2
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.l3 = nn.Conv3d(512, 256, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

        # 3
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.l4 = nn.Conv3d(256, 128, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l5 = nn.Conv3d(128, 128, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

        # 4
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.l6 = nn.Conv3d(128, 64, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l7 = nn.Conv3d(64, 64, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

        # 5
        self.up4 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.l8 = nn.Conv3d(64, 1, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l9 = nn.Conv3d(1, 1, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

    def forward(self, inputs):
        x = self.activation(self.l1(inputs))
        x = x.view(-1, 512, self.depth, 8, 8)
        x = self.activation(self.up0(x))
        x = self.activation(self.l2(x))

        x = self.activation(self.up1(x))
        x = self.activation(self.l3(x))

        x = self.activation(self.up2(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.l5(x))

        x = self.activation(self.up3(x))
        x = self.activation(self.l6(x))
        x = self.activation(self.l7(x))

        x = self.activation(self.up4(x))
        x = self.activation(self.l8(x))
        x = self.l9(x)

        return x


# 3D AutoEncoder
class TdAE(nn.Module):
    def __init__(self, in_seq_size, out_seq_size, channel_size, hidden_size):
        super(TdAE, self).__init__()

        self.in_seq_size = in_seq_size
        self.out_seq_size = out_seq_size
        # self.channel_size = channel_size
        self.hidden_size = hidden_size

        self.encoder = TdEncoder(in_seq_size, 1, hidden_size)
        self.decoder = TdDecoder(out_seq_size, 1, hidden_size)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# TODO Loss
class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)