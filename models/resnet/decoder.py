from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.decoder(x)
        return x
