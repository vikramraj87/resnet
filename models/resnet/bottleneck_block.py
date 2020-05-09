from torch import nn
from utils.conv_bn import ConvBatchNorm


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 down_sampling=1,
                 activation=nn.ReLU):
        super().__init__()

        self.expanded_channels = out_channels * self.expansion
        self.activation = activation()
        self.conv_bn1 = ConvBatchNorm(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)
        self.conv_bn2 = ConvBatchNorm(out_channels,
                                      out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      stride=down_sampling,
                                      bias=False)
        self.conv_bn3 = ConvBatchNorm(out_channels,
                                      self.expanded_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=False)

    def forward(self, x):
        x = self.activation(self.conv_bn1(x))
        x = self.activation(self.conv_bn2(x))
        x = self.conv_bn3(x)

        return x
