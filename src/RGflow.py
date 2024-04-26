import torch
import torch.nn as nn
import torch.nn.functional as F


# RG include RG4,RG3,RG2,RG1
class RG4(nn.Module):
    def __init__(self, channels, out):
        super(RG4, self).__init__()
        self.conv = nn.Conv2d(channels//5, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.ru = nn.ReLU(out)
        self.es_4_Conv = nn.Conv2d(512+256, 256, kernel_size=3, padding=1)
        self.rg_4_Conv = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)  # dilated

    def forward(self, p1, p2, p3):  # p1=encoder, p2=sam, p3=decoder
        g4 = torch.cat((p1, p2), dim=1)
        g4 = F.relu(self.es_4_Conv(g4))
        d5 = -1*torch.sigmoid(p3)+1                              # reverse
        r4 = self.rg_4_Conv(d5)                                  # 16*16 ->32*32
        rg4 = torch.mul(g4, r4)

        rg4 = self.ru(self.bn(self.conv(rg4)))
        return rg4


class RG3(nn.Module):
    def __init__(self, channels, out):
        super(RG3, self).__init__()
        self.conv = nn.Conv2d(channels//5, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.ru = nn.ReLU(out)
        self.es_3_Conv = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.rg_3_Conv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)  # dilated

    def forward(self, p1, p2, p3):  # p1=encoder, p2=sam, p3=decoder
        g3 = torch.cat((p1, p2), dim=1)
        g3 = F.relu(self.es_3_Conv(g3))
        d4 = -1*torch.sigmoid(p3)+1                              # reverse
        r3 = self.rg_3_Conv(d4)                                  # 32*32 ->64*64
        rg3 = torch.mul(g3, r3)

        rg3 = self.ru(self.bn(self.conv(rg3)))
        return rg3


class RG2(nn.Module):
    def __init__(self, channels, out):
        super(RG2, self).__init__()
        self.conv = nn.Conv2d(channels//5, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.ru = nn.ReLU(out)
        self.es_2_Conv = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
        self.rg_2_Conv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)  # dilated

    def forward(self, p1, p2, p3):  # p1=encoder, p2=sam, p3=decoder
        # p2 = F.interpolate(p2, scale_factor=4, mode='bilinear')  # 64*64 ->128*128
        g2 = torch.cat((p1, p2), dim=1)
        g2 = F.relu(self.es_2_Conv(g2))
        d3 = -1*torch.sigmoid(p3)+1                              # reverse
        r2 = self.rg_2_Conv(d3)                                  # 64*64 ->128*128
        rg2 = torch.mul(g2, r2)

        rg2 = self.ru(self.bn(self.conv(rg2)))
        return rg2


class RG1(nn.Module):
    def __init__(self, channels, out):
        super(RG1, self).__init__()
        self.conv = nn.Conv2d(channels//5, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.ru = nn.ReLU(out)
        self.es_1_Conv = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
        self.rg_1_Conv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)  # dilated

    def forward(self, p1, p2, p3):  # p1=encoder, p2=sam, p3=decoder
        g1 = torch.cat((p1, p2), dim=1)
        g1 = F.relu(self.es_1_Conv(g1))
        d2 = -1*torch.sigmoid(p3)+1                              # reverse
        r1 = self.rg_1_Conv(d2)                                  # 128*128 ->256*256
        rg1 = torch.mul(g1, r1)

        rg1 = self.ru(self.bn(self.conv(rg1)))
        return rg1





















