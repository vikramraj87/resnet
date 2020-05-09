from torch import nn
from .basic_block import BasicBlock
from .basic_shortcut import BasicShortcut
from .unit import Unit


class Layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block=BasicBlock,
                 shortcut=BasicShortcut,
                 n=1,
                 activation=nn.ReLU):
        super().__init__()

        down_sampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            Unit(in_channels,
                 out_channels,
                 block=block,
                 shortcut=shortcut,
                 down_sampling=down_sampling,
                 activation=activation),
            *[Unit(out_channels * block.expansion,
                   out_channels,
                   block=block,
                   shortcut=shortcut,
                   activation=activation) for _ in range(n - 1)])

    def forward(self, x):
        x = self.blocks(x)
        return x

    @property
    def out_features(self):
        return self.blocks[-1].out_features
