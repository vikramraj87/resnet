from torch import nn
from .basic_block import BasicBlock
from .basic_shortcut import BasicShortcut


class Unit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block=BasicBlock,
                 shortcut=BasicShortcut,
                 down_sampling=1,
                 activation=nn.ReLU):
        super().__init__()
        self.block = block(in_channels,
                           out_channels,
                           down_sampling,
                           activation)
        self.shortcut = shortcut(in_channels,
                                 out_channels,
                                 block.expansion,
                                 down_sampling)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        return x

    @property
    def out_features(self):
        return self.block.expanded_channels
