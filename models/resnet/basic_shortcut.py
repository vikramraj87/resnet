from torch import nn
from utils.conv_bn import ConvBatchNorm


class BasicShortcut(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 down_sampling=1):
        super().__init__()

        expanded_channels = out_channels * expansion

        if in_channels != expanded_channels:
            self.model = ConvBatchNorm(in_channels,
                                       expanded_channels,
                                       kernel_size=1,
                                       stride=down_sampling,
                                       padding=0,
                                       bias=False)
        else:
            self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)

