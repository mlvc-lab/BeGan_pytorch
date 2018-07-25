import torch.nn as nn
import torch.nn.functional as F


# 3D Encoder
class TdEncoder(nn.Module):
    def __init__(self, seq_size, input_channel, hidden_size,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1, activation=F.elu):
        super(TdEncoder, self).__init__()

        self.seq_size = seq_size
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation

        # TODO change kernel size to figure out long-term and short-term effect
        # 64x64xchan
        self.l1 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l2 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.down1 = nn.Conv3d(self.seq_size, self.seq_size, 1,
                               stride=self.stride, padding=0, groups=self.seq_size)
        self.pool1 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        # 32x32xchan
        self.l3 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l4 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.down2 = nn.Conv3d(self.seq_size, self.seq_size * 2, 1,
                               stride=self.stride, padding=0, groups=self.seq_size)
        self.pool2 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        # 16x16x(chan x 2)
        self.l5 = nn.Conv3d(self.seq_size * 2, self.seq_size * 2, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l6 = nn.Conv3d(self.seq_size * 2, self.seq_size * 2, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.down3 = nn.Conv3d(self.seq_size * 2, self.seq_size * 3, 1,
                               stride=self.stride, padding=0, groups=self.seq_size)
        self.pool3 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        # 8x8x(chan x 3)
        self.l7 = nn.Conv3d(self.seq_size * 3, self.seq_size * 3, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l8 = nn.Conv3d(self.seq_size * 3, self.seq_size * 3, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l9 = nn.Linear((self.seq_size * 3) * self.input_channel * 8 * 8, self.hidden_size)

    def forward(self, inputs):
        print('in', inputs.size())
        
        x = self.activation(self.l1(inputs))
        x = self.activation(self.l2(x))
        x = self.activation(self.down1(x))
        x = self.activation(self.pool1(x))

        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.down2(x))
        x = self.activation(self.pool2(x))

        x = self.activation(self.l5(x))
        x = self.activation(self.l6(x))
        x = self.activation(self.down3(x))
        x = self.activation(self.pool3(x))

        x = self.activation(self.l7(x))
        x = self.activation(self.l8(x))
        x = x.view(-1, (self.seq_size * 3) * self.input_channel * 8 * 8)
        x = self.activation(self.l9(x))

        return x


# 3D Decoder
class TdDecoder(nn.Module):
    def __init__(self, seq_size, input_channel, hidden_size,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1, activation=F.elu):
        super(TdDecoder, self).__init__()

        self.seq_size = seq_size
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation = activation

        # TODO change kernel size to figure out long-term and short-term effect
        # 8x8xchan
        self.l1 = nn.Linear(self.hidden_size, (self.seq_size * self.input_channel * 8 * 8))
        self.l2 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l3 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)

        # 16x16xchan
        self.up1 = nn.Upsample(scale_factor=2)
        self.l4 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l5 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=(2, 1, 1), padding=self.padding, groups=self.groups)

        # 32x32xchan
        self.up2 = nn.Upsample(scale_factor=2)
        self.l6 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l7 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=(2, 1, 1), padding=self.padding, groups=self.groups)

        # 64x64xchan
        self.up3 = nn.Upsample(scale_factor=2)
        self.l8 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l9 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.groups)
        self.l10 = nn.Conv3d(self.seq_size, self.seq_size, self.kernel_size,
                             stride=(2, 1, 1), padding=self.padding, groups=self.groups)

    def forward(self, inputs):
        
        print(inputs.size())
        
        x = self.activation(self.l1(inputs))
        x = x.view(-1, self.seq_size, self.input_channel, 8, 8)
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        print(x.size())

        x = self.activation(self.up1(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.l5(x))

        x = self.activation(self.up2(x))
        x = self.activation(self.l6(x))
        x = self.activation(self.l7(x))

        x = self.activation(self.up3(x))
        x = self.activation(self.l8(x))
        x = self.activation(self.l9(x))
        x = self.activation(self.l10(x))

        return x
    
# TODO Loss
