import torch
import torch.nn as nn
import torch.nn.functional as F

from ANet.src.MRA import *
from ANet.src.RGflow import *
from ANet.src.SGflow import *


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicConv2d(in_channels, in_channels // 4)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
            self.conv = BasicConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicConv2d(in_channels, in_channels // 5)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 5, kernel_size=2, stride=2)
            self.conv = BasicConv2d(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SAMConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, 32)
        self.conv2 = BasicConv2d(32, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MGNet(nn.Module):
    def __init__(self, in_channels, seg_prior, boundary_prior, n_class):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.seg_prior = seg_prior
        self.boundary_prior = boundary_prior
        self.n_class = n_class

        self.samin = OutConv(7, in_channels)
        self.encoder1 = BasicConv2d(in_channels, 64)
        self.M1 = MDA(64)
        self.encoder2 = Down(64, 128)
        self.M2 = MDA(128)
        self.encoder3 = Down(128, 256)
        self.M3 = MDA(256)
        self.encoder4 = Down(256, 512)
        self.M4 = MDA(512)
        self.encoder5 = Down(512, 512)

        self.MM1 = RFB_modified(64, 16)
        self.MM2 = RFB_modified(128, 16)
        self.MM3 = RFB_modified(256, 16)
        self.MM4 = RFB_modified(512, 16)

        self.MM = MMA(64, 128)
        self.Rs = RS(640, 512)

        # train
        self.ap = OutConv(3, 32)
        self.cp = SAMConv(1, 32)

        self.SAM = OutConv(1154, 1024)

        # val
        self.smx = OutConv(1152, 1024)

        self.sam4 = OutConv(1024, 256)
        self.sam3 = OutConv(1024, 128)
        self.sam2 = OutConv(1024, 64)
        self.sam1 = OutConv(1024, 32)

        self.rg4 = RG4(1280, 512)
        self.rg3 = RG3(640, 256)
        self.rg2 = RG2(320, 128)
        self.rg1 = RG1(160, 64)

        self.sg3 = SG3(512, 128)
        self.sg2 = SG2(256, 64)
        self.sg1 = SG1(128, 32)

        self.decoder5 = BasicConv2d(1024, 512)
        self.decoder4 = Up(512+512, 256)
        self.decoder3 = Up2(256+256+128, 128)
        self.decoder2 = Up2(128+128+64, 64)
        self.decoder1 = Up2(64+64+32, 32)
        self.out = OutConv(5*n_class, n_class)

        self.deep_5 = OutConv(512, n_class)
        self.deep_4 = OutConv(256, n_class)
        self.deep_3 = OutConv(128, n_class)
        self.deep_2 = OutConv(64, n_class)
        self.deep_1 = OutConv(32, n_class)

    def forward(self, x, y=None, z=None):

        if self.training and y is not None and z is not None:

            x0 = torch.cat((x, y, z), dim=1)
            x = self.samin(x0)
        else:
            x = x

        x = self.encoder1(x)
        m1 = self.M1(x)
        x = self.encoder2(x)
        m2 = self.M2(x)
        x = self.encoder3(x)
        m3 = self.M3(x)
        x = self.encoder4(x)
        m4 = self.M4(x)
        x = self.encoder5(x)

        mm1 = self.MM1(m1)
        mm2 = self.MM2(m2)
        mm3 = self.MM3(m3)
        mm4 = self.MM4(m4)
        mma = self.MM(mm1, mm2, mm3, mm4)
        rs = self.Rs(x, mma)
        s = torch.cat((rs, x, mma), dim=1)

        sam = self.smx(s)
        sam4 = F.interpolate(sam, scale_factor=2, mode='bilinear')
        sam4 = self.sam4(sam4)
        sam3 = F.interpolate(sam, scale_factor=4, mode='bilinear')
        sam3 = self.sam3(sam3)
        sam2 = F.interpolate(sam, scale_factor=8, mode='bilinear')
        sam2 = self.sam2(sam2)
        sam1 = F.interpolate(sam, scale_factor=16, mode='bilinear')
        sam1 = self.sam1(sam1)

        xd5 = self.decoder5(sam)
        deep5 = F.interpolate(xd5, scale_factor=16, mode='bilinear')
        deepmap5 = self.deep_5(deep5)  # Deep Supervision
        S3 = self.sg3(xd5)

        RG_4 = self.rg4(m4, sam4, xd5)
        xd4 = self.decoder4(xd5, RG_4)
        deep4 = F.interpolate(xd4, scale_factor=8, mode='bilinear')
        deepmap4 = self.deep_4(deep4)  # Deep Supervision
        S2 = self.sg2(xd4)

        RG_3 = self.rg3(m3, sam3, xd4)
        xd3 = self.decoder3(xd4, RG_3, S3)
        deep3 = F.interpolate(xd3, scale_factor=4, mode='bilinear')
        deepmap3 = self.deep_3(deep3)  # Deep Supervision
        S1 = self.sg1(xd3)

        RG_2 = self.rg2(m2, sam2, xd3)
        xd2 = self.decoder2(xd3, RG_2, S2)
        deep2 = F.interpolate(xd2, scale_factor=2, mode='bilinear')
        deepmap2 = self.deep_2(deep2)  # Deep Supervision

        RG_1 = self.rg1(m1, sam1, xd2)
        xd1 = self.decoder1(xd2, RG_1, S1)
        deepmap1 = self.deep_1(xd1)  # Deep Supervision

        deepmap0 = self.out(torch.cat((deepmap1, deepmap2, deepmap3, deepmap4, deepmap5),1))

        # return self.sigmoid(deepmap0)  # Use this for testing

        return torch.sigmoid(deepmap0), torch.sigmoid(deepmap1), torch.sigmoid(deepmap2), torch.sigmoid(deepmap3), \
            torch.sigmoid(deepmap4), torch.sigmoid(deepmap5)


if __name__ == '__main__':
    model = MGNet(in_channels=3, seg_prior=3, boundary_prior=1, n_class=2).cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    seg_tensor = torch.randn(1, 3, 256, 256).cuda()
    bou_tensor = torch.randn(1, 1, 256, 256).cuda()
    out = model(input_tensor, seg_tensor, bou_tensor)
    print(out[0].shape)


