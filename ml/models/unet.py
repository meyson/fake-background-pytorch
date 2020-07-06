import torch

from .base import *


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv = DoubleConv(in_c, out_c)
        self.max_pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.double_conv(x)
        pool_out = self.max_pool2x2(conv_out)
        return conv_out, pool_out


def center_crop_tensor(tensor, delta):
    return tensor[:, :, delta:-delta, delta:-delta]


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2
        )

    def forward(self, x, y):
        up = self.transpose_conv(y)
        delta = (x.shape[2] - up.shape[2]) // 2

        x = center_crop_tensor(x, delta)
        cat = torch.cat((x, up), dim=1)
        res = self.conv(cat)
        return res


class UNet(SegmentationModel):
    def __init__(self, pretrained=False, in_c=1):
        super().__init__(pretrained)

        if self.pretrained:
            raise NotImplementedError('Pretrained weights not available.')

        self.down1 = Down(in_c, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottom = DoubleConv(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x1, x1_max = self.down1(x)
        x2, x2_max = self.down2(x1_max)
        x3, x3_max = self.down3(x2_max)
        x4, x4_max = self.down4(x3_max)

        x = self.bottom(x4_max)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = UNet(in_c=1)
    batch = torch.rand(1, 1, 572, 572)
    logits = model(batch)
    print(logits.shape)
