from torch import nn
from .basic_block import BasicBlock
from .basic_shortcut import BasicShortcut
from .bottleneck_block import BottleneckBlock
from .decoder import Decoder
from .encoder import Encoder


class Model(nn.Module):
    def __init__(self,
                 in_channels,
                 n_classes,
                 block_sizes=(64, 128, 256, 512),
                 depths=(2, 2, 2, 2),
                 activation=nn.ReLU,
                 block=BasicBlock,
                 shortcut=BasicShortcut):
        super().__init__()

        self.encoder = Encoder(in_channels,
                               block_sizes=block_sizes,
                               depths=depths,
                               activation=activation,
                               block=block,
                               shortcut=shortcut)

        decoder_in_features = self.encoder.out_features
        self.decoder = Decoder(decoder_in_features, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes):
    return Model(in_channels, n_classes)


def resnet34(in_channels, n_classes):
    return Model(in_channels, n_classes, depths=(3, 4, 6, 3))


def resnet50(in_channels, n_classes):
    return Model(in_channels,
                 n_classes,
                 depths=(3, 4, 6, 3),
                 block=BottleneckBlock)


def resnet101(in_channels, n_classes):
    return Model(in_channels,
                 n_classes,
                 depths=(3, 4, 23, 3),
                 block=BottleneckBlock)


def resnet152(in_channels, n_classes):
    return Model(in_channels,
                 n_classes,
                 depths=(3, 8, 36, 3),
                 block=BottleneckBlock)
