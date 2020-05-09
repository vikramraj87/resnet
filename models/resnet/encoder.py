from torch import nn
from .basic_block import BasicBlock
from .basic_shortcut import BasicShortcut
from .layer import Layer
from utils.conv_bn import ConvBatchNorm


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 block_sizes=(64, 128, 256, 512),
                 depths=(2, 2, 2, 2),
                 activation=nn.ReLU,
                 block=BasicBlock,
                 shortcut=BasicShortcut):
        super().__init__()

        self.gate = nn.Sequential(
            ConvBatchNorm(in_channels,
                          block_sizes[0],
                          kernel_size=7,
                          stride=2,
                          padding=3),
            activation(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1)
        )
        in_out = list(zip(block_sizes, block_sizes[1:]))
        self.blocks = nn.ModuleList([
            Layer(block_sizes[0],
                  block_sizes[0],
                  n=depths[0],
                  activation=activation,
                  block=block,
                  shortcut=shortcut),
            *[Layer(in_c * block.expansion,
                    out_c,
                    n=n,
                    activation=activation,
                    block=block,
                    shortcut=shortcut)
              for (in_c, out_c), n in zip(in_out, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def out_features(self):
        return self.blocks[-1].out_features
