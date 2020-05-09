from torch import nn
from utils.conv_bn import ConvBatchNorm


class BasicBlock(nn.Module):
    expansion = 1

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
                                      kernel_size=3,
                                      stride=down_sampling,
                                      padding=1)
        self.conv_bn2 = ConvBatchNorm(out_channels,
                                      self.expanded_channels,
                                      kernel_size=3,
                                      padding=1)

    def forward(self, x):
        x = self.conv_bn1(x)
        x = self.activation(x)
        x = self.conv_bn2(x)
        return x
