import torch
import torch.nn as nn


class SG3(nn.Module):
    def __init__(self, channels, out):   # 512 --->128
        super(SG3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sg_conv3 = nn.Conv2d(channels, out, 3, padding=1)

    def forward(self, x):
        sg = self.upsample(x)
        sg = self.sg_conv3(sg)
        sg = torch.sigmoid(sg)

        return sg


class SG2(nn.Module):
    def __init__(self, channels, out):
        super(SG2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sg_conv3 = nn.Conv2d(channels, out, 3, padding=1)

    def forward(self, x):
        sg = torch.sigmoid(x)
        sg = self.upsample(sg)
        sg = self.sg_conv3(sg)
        return sg


class SG1(nn.Module):
    def __init__(self, channels,out):
        super(SG1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sg_conv3 = nn.Conv2d(channels, out, 3, padding=1)

    def forward(self, x):
        sg = torch.sigmoid(x)
        sg = self.upsample(sg)
        sg = self.sg_conv3(sg)
        return sg
