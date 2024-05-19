from .cbam import CBAM
from .SNL import Resblock
import torch.nn as nn
import torch

class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

class Enhanced_multi_atten(nn.Module):
    def __init__(self, channel):
        super(Enhanced_multi_atten, self).__init__()

        self.CBAM = CBAM(channel)

        self.SNL = Resblock(channel)

        self.conv= nn.Conv2d(channel*2, channel, 1)
        self.BN=_BatchNorm2d(channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):  # x:[1, 256, 31, 31]
        x1 = self.CBAM(x)  #[1, 256, 31, 31]

        x2 = self.SNL(x)  #[1, 256, 31, 31]

        out=self.conv(torch.cat((x1, x2), 1))
        out=self.BN(out)
        out=x+self.sigmoid(out)

        return out