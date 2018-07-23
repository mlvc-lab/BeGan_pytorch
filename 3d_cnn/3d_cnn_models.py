import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.modules):
    def __init__(self,batch_size, input_size, input_channel, hidden_size, 
                 output_channel=None, kernel_size=(3, 3), stride=1, padding=0, group=None, activation=F.elu(True)):
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.output_channel = output_channel if output_channel is not None else input_channel * 3
        self.kernel_size = (kernel_size[0], kernel_size[1], input_channel)
        self.stride = stride
        self.padding = padding
        self.group = group if group is not None else input_channel
        self.activation = activation
        
        # TODO change kernel size to figure out long-term and short-term effect
        # 64x64xchan
        self.l1 = nn.Conv3d(self.input_channel, self.input_channel, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.l2 = nn.Conv3d(self.input_channel, self.input_channel, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.down1 = nn.Conv3d(self.input_channel, self.input_channel, (1, 1, 1),
                               stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.pool1 = nn.AvgPool3d(2, 2)

        # 32x32xchan
        self.l3 = nn.Conv3d(self.input_channel, self.input_channel, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.l4 = nn.Conv3d(self.input_channel, self.input_channel, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.down2 = nn.Conv3d(self.input_channel, self.input_channel * 2, (1, 1, 1),
                               stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.pool2 = nn.AvgPool3d(2, 2)
        
        # 16x16x(chan x 2)
        self.l5 = nn.Conv3d(self.input_channel * 2, self.input_channel * 2, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.l6 = nn.Conv3d(self.input_channel * 2, self.input_channel * 2, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.down3 = nn.Conv3d(self.input_channel * 2, self.input_channel * 3, (1, 1, 1),
                               stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.pool3 = nn.AvgPool3d(2, 2)
        
        # 8x8x(chan x 3)
        self.l7 = nn.Conv3d(self.input_channel * 3, self.input_channel * 3, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.l8 = nn.Conv3d(self.input_channel * 3, self.input_channel * 3, self.kernel_size,
                            stride=self.stride, padding=self.padding, groups=self.input_channel)
        self.l9 = nn.Linear(8 * 8 * output_channel, self.hidden_size)
        
    def forward(self, inputs):
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
        x = x.view(-1, 8 * 8 * self.output_channel)
        x = self.activation(self.l9(x))
        
        return x
    
# TODO Decoder

# TODO Loss
