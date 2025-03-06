"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''Conv2d -> BatchNorm -> LeakyReLU -> Dropout) * 2'''
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    '''Upsampling followed by DoubleConv'''
    def __init__(self, in_channels, out_channels, skip_channels):
        super(Up, self).__init__()
        # Total input channels: channels from upsampled layer + all skip connections
        total_in_channels = in_channels + skip_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(total_in_channels, out_channels)

    def forward(self, x1, *x_skip):
        x1 = self.up(x1)
        if x_skip:
            x1 = torch.cat([x1] + list(x_skip), dim=1)
        return self.conv(x1)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=16):
        super(UNetPlusPlus, self).__init__()

        # Downsampling paths
        self.x00 = DoubleConv(in_channels, base_filters * 2)
        self.x10 = DoubleConv(base_filters * 2, base_filters * 4)
        self.x20 = DoubleConv(base_filters * 4, base_filters * 8)
        self.x30 = DoubleConv(base_filters * 8, base_filters * 16)
        self.x40 = DoubleConv(base_filters * 16, base_filters * 32)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling paths with dynamic skip channel sizes
        self.x01 = Up(base_filters * 4, base_filters * 2, skip_channels=base_filters * 2)  # x10 + x00
        self.x11 = Up(base_filters * 8, base_filters * 4, skip_channels=base_filters * 4)  # x20 + x10
        self.x21 = Up(base_filters * 16, base_filters * 8, skip_channels=base_filters * 8)  # x30 + x20
        self.x02 = Up(base_filters * 4, base_filters * 2, skip_channels=base_filters * 4)  # x11 + x01
        self.x12 = Up(base_filters * 8, base_filters * 4, skip_channels=base_filters * 8)  # x21 + x11
        self.x03 = Up(base_filters * 2, base_filters * 2, skip_channels=base_filters * 6)  # x12 + x01 + x02

        # Final output layer
        self.final = nn.Conv2d(base_filters * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x00 = self.x00(x)
        p0 = self.pool(x00)

        x10 = self.x10(p0)
        p1 = self.pool(x10)

        x20 = self.x20(p1)
        p2 = self.pool(x20)

        x30 = self.x30(p2)
        p3 = self.pool(x30)

        x40 = self.x40(p3)

        # Upsampling
        x01 = self.x01(x10, x00)
        x11 = self.x11(x20, x10)
        x02 = self.x02(x11, x01)
        x21 = self.x21(x30, x20)
        x12 = self.x12(x21, x11)
        x03 = self.x03(x12, x01, x02)

        # Final output
        output = self.final(x03)
        return torch.softmax(output, dim=1)

# Define parameters
in_channels = 3  # e.g., RGB images
out_channels = 2  # e.g., 2 classes
height, width = 256, 256  # Input image dimensions

# Create model and input tensor
model = UNetPlusPlus(in_channels, out_channels)
input_tensor = torch.randn(1, in_channels, height, width)  # Batch size = 1

# Perform a forward pass
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # After concatenation, the number of channels is (out_channels + skip_connection_channels)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2=None):
        # Upsample x1
        x1 = self.up(x1)
        # Align and concatenate with x2 (skip connection)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class NestedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, deep_supervision=False):
        super(NestedUNet, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [64, 128, 256, 512, 1024]

        # Downsampling layers
        self.inc = DoubleConv(in_channels, filters[0])
        self.down1 = nn.MaxPool2d(2)
        self.conv1_0 = DoubleConv(filters[0], filters[1])
        self.down2 = nn.MaxPool2d(2)
        self.conv2_0 = DoubleConv(filters[1], filters[2])
        self.down3 = nn.MaxPool2d(2)
        self.conv3_0 = DoubleConv(filters[2], filters[3])
        self.down4 = nn.MaxPool2d(2)
        self.conv4_0 = DoubleConv(filters[3], filters[4])

        # Nested Upsampling Layers
        self.up0_1 = Up(filters[1], filters[0], bilinear)
        self.up1_1 = Up(filters[2], filters[1], bilinear)
        self.up0_2 = Up(filters[1], filters[0], bilinear)
        self.up2_1 = Up(filters[3], filters[2], bilinear)
        self.up1_2 = Up(filters[2], filters[1], bilinear)
        self.up0_3 = Up(filters[1], filters[0], bilinear)
        self.up3_1 = Up(filters[4], filters[3], bilinear)
        self.up2_2 = Up(filters[3], filters[2], bilinear)
        self.up1_3 = Up(filters[2], filters[1], bilinear)
        self.up0_4 = Up(filters[1], filters[0], bilinear)


        # Final layers
        if deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x0_0 = self.inc(x)
        x1_0 = self.conv1_0(self.down1(x0_0))
        x2_0 = self.conv2_0(self.down2(x1_0))
        x3_0 = self.conv3_0(self.down3(x2_0))
        x4_0 = self.conv4_0(self.down4(x3_0))

        # Nested upsampling
        x0_1 = self.up0_1(x1_0, x0_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x0_2 = self.up0_2(x1_1, x0_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x1_2 = self.up1_2(x2_1, x1_0)
        x0_3 = self.up0_3(x1_2, x0_0)
        x3_1 = self.up3_1(x4_0, x3_0)
        x2_2 = self.up2_2(x3_1, x2_0)
        x1_3 = self.up1_3(x2_2, x1_0)
        x0_4 = self.up0_4(x1_3, x0_0)

        # Final output
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            out = self.final(x0_4)
            return out
        
in_channels = 3  # For RGB input images
out_channels = 1  # For single-channel output (e.g., binary segmentation)
bilinear = False
deep_supervision = False

# Instantiate the model
model = NestedUNet(in_channels, out_channels, bilinear=bilinear, deep_supervision=deep_supervision)

input_tensor = torch.randn(1, 3, 256, 256)

# Perform a forward pass
output = model(input_tensor)

# Check the output shape
if deep_supervision:
    for i, out in enumerate(output):
        print(f"Output {i + 1} shape: {out.shape}")
else:
    print(f"Output shape: {output.shape}") # Should be [batch_size, out_channels, height, width]