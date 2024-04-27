import torch
import torch.nn as nn
import torch.nn.functional as F


class MDA(nn.Module):
    def __init__(self, in_ch):
        super(MDA, self).__init__()
        self.APool_W = nn.AdaptiveAvgPool2d((None, 1))
        self.APool_H = nn.AdaptiveAvgPool2d((None, 1))
        self.APool_C = nn.AdaptiveAvgPool2d((None, 1))
        self.InConv = nn.Conv2d(in_ch, in_ch // 4, kernel_size=1)
        self.OuConv = nn.Conv2d(in_ch // 4, in_ch, kernel_size=1)
        self.detal = nn.Parameter(torch.tensor([0.1]))


    def forward(self, x):
        input = self.InConv(x)
        B, C, H, W, = input.size()

        # W_attention
        input_w = self.APool_W(input)  # [B C H 1]
        f_w = F.softmax(input_w, dim=3).permute(0, 3, 1, 2).view(B, -1, 1)  # [B, C*H, 1]
        G_w = torch.mul(input.view(B, C*H, W), f_w)  # [B, C*H, W]
        T_w = F.softmax(torch.matmul(G_w.permute(0, 2, 1), G_w), dim=1)  # [B, W, W]
        Wo = torch.matmul(input.view(B, C*H, W), T_w).view(B, C, H, W)  # [B, C, H, W]

        # H_attention
        input_h = self.APool_H(input.permute(0, 1, 3, 2))  # [B C W 1]
        f_h = F.softmax(input_h, dim=3).permute(0, 3, 1, 2).view(B, C*W, 1)
        G_h = torch.mul(input.view(B, C*W, H), f_h)  # [B, C*W, H]
        T_h = F.softmax(torch.matmul(G_h.permute(0, 2, 1), G_h), dim=1)  # [B, H, H]
        Ho = torch.matmul(input.view(B, C*W, H), T_h).view(B, C, H, W)

        # C_attention
        input_c = self.APool_C(input.permute(0, 2, 3, 1))  # [B H W 1]
        f_c = F.softmax(input_c, dim=3).permute(0, 3, 1, 2).view(B, H * W, 1)
        G_c = torch.mul(f_c, input.permute(0, 2, 3, 1).view(B, H*W, C))  # [B, H*W, C ]
        T_c = F.softmax(torch.matmul(G_c.permute(0, 2, 1), G_c), dim=1)  # [B, C, C]
        Co = torch.matmul(input.permute(0, 2, 3, 1).view(B, H*W, C), T_c).view(B, C, H, W)

        # CHW_attention
        out = self.OuConv(self.detal*(Co+Wo+Ho))+x
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_ch, ou_ch, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, ou_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(ou_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MMA(nn.Module):
    def __init__(self, mid_ch, out_ch):   # 256, 512
        super(MMA, self).__init__()
        self.relu = nn.ReLU(True)

        self.dpsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dpsample_1 = BasicConv2d(mid_ch//4, mid_ch//4, 3, padding=1)
        self.dpsample_2 = BasicConv2d(mid_ch//4, mid_ch//4, 3, padding=1)
        self.dpsample_3 = BasicConv2d(mid_ch//4, mid_ch//4, 3, padding=1)
        self.dpsample_4 = BasicConv2d(mid_ch//4, mid_ch//4, 3, padding=1)
        self.dpsample_5 = BasicConv2d(mid_ch//4, mid_ch//4, 3, padding=1)

        self.cat2 = BasicConv2d(mid_ch//2, mid_ch//4, 3, padding=1)
        self.cat3 = BasicConv2d(mid_ch//2, mid_ch//4, 3, padding=1)
        self.cat4 = BasicConv2d(mid_ch//2, mid_ch, 3, padding=1)

        self.cat5 = BasicConv2d(mid_ch//2, mid_ch*2, 3, padding=1)
        self.outconv = nn.Conv2d(mid_ch*2, out_ch, 1)

    def forward(self, x1, x2, x3, x4):   # x1:256*256*256, x2:256*128*128, x3:256*64*64, x4:256*32*32
        x1_1 = x1

        x2_1 = self.dpsample_1(self.dpsample(x1)) * x2
        x3_1 = self.dpsample_2(self.dpsample(self.dpsample(x1))) * self.dpsample_3(self.dpsample(x2)) * x3
        x4_1 = self.dpsample_3(self.dpsample(self.dpsample(self.dpsample(x1)))) \
               * self.dpsample_4(self.dpsample(self.dpsample(x2))) * self.dpsample_5(self.dpsample(x3)) * x4

        x2_2 = torch.cat((x2_1, self.dpsample_2(self.dpsample(x1_1))), dim=1)  # 64+64
        x2_2 = self.cat2(x2_2)

        x3_2 = torch.cat((x3_1, self.dpsample_3(self.dpsample(x2_2))), dim=1)  # 64+64
        x3_2 = self.cat3(x3_2)

        x4_2 = torch.cat((x4_1, self.dpsample_4(self.dpsample(x3_2))), dim=1)  # 64+64
        x4_2 = self.dpsample(x4_2)
        x5_2 = self.cat5(x4_2)

        x = self.outconv(x5_2)
        return x


class RS(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(RS, self).__init__()
        self.Conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.Convm = nn.Conv2d(in_ch//5, 1, kernel_size=1)
        self.OCon = nn.Conv2d(ou_ch, ou_ch, kernel_size=1)
        self.detal = nn.Parameter(torch.tensor([0.1]))

    def forward(self, e5, fmma):
        emma = torch.cat((e5, fmma), dim=1)
        ex = torch.sigmoid(self.Conv(emma))
        ex = torch.matmul(e5, ex)*self.detal + e5  # element-wise addition
        mx = torch.sigmoid(self.Convm(fmma))
        x = torch.matmul(ex, mx)
        x = self.OCon(x)
        return x*self.detal + x

















