"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    '''Upscaling then double conv'''
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Planet(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, bilinear=False):
        super(Planet, self).__init__()
        self.bilinear = bilinear

        no_layer = 0
        inp_size = height
        start_filter = 4
        while inp_size > 8:
            no_layer += 1
            inp_size = inp_size // 2

        # Encoder
        self.encoder = nn.ModuleList()
        self.inc = DoubleConv(in_channels, start_filter)
        self.encoder.append(self.inc)
        for i in range(1, no_layer):
            start_filter *= 2
            self.encoder.append(Down(start_filter // 2, start_filter))

        # Middle layer
        self.mid = DoubleConv(start_filter, start_filter * 2)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(no_layer):
            self.decoder.append(Up(start_filter * 2, start_filter // 2, bilinear))
            start_filter //= 2

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x_list = []
        for layer in self.encoder:
            x = layer(x)
            x_list.append(x)
        x = self.mid(x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x, x_list[-(i+1)])
        logits = self.outc(x)
        return torch.tanh(logits)
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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
"""
class Up(nn.Module):
    '''Upscaling then double conv'''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # Bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Adjust input channels for DoubleConv
        self.conv = DoubleConv(in_channels, out_channels)  # Note: in_channels is doubled after concatenation

    def forward(self, x1, x2):
        # Upsample x1
        x1 = self.up(x1)

        # Adjust spatial dimensions of x1 to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
"""

class Up(nn.Module):
    '''Upscaling then double conv'''
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()

        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc= nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.tanh(logits)
    
class UNetPlusPlus(nn.Module): 
    def __init__(self, in_channels, out_channels, bilinear=False): 
        super(UNetPlusPlus, self).__init__() 
        self.bilinear = bilinear 
        
        self.inc = DoubleConv(in_channels, 32) 
        self.down1 = Down(32, 64) 
        self.down2 = Down(64, 128) 
        self.down3 = Down(128, 256) 
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256) 
        self.up2 = Up(256, 128) 
        self.up3 = Up(128, 64) 
        self.up4 = Up(64, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1) 

    def forward(self, x): 
        x00 = self.inc(x) 
        x10 = self.down1(x00) 
        x20 = self.down2(x10) 
        x30 = self.down3(x20) 
        x40 = self.down4(x30)

        x01 = self.up1(x10, x00) 
        x11 = self.up2(x20, x10) 
        x21 = self.up3(x30, x20) 
        x31 = self.up4(x40, x30)

        x02 = self.up1(x11, x01) 
        x12 = self.up2(x21, x11) 
        x22 = self.up3(x31, x21)

        x03 = self.up1(x12, x02) 
        x13 = self.up2(x22, x12)

        x04 = self.up1(x13, x03)

        logits = self.outc(x04) 
        return torch.sigmoid(logits)
    
class Planet(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, bilinear=False):
        super(Planet, self).__init__()
        self.bilinear = bilinear

        # Calculate number of layers based on height
        no_layer = 0
        inp_size = height
        start_filter = 4
        while inp_size > 8:
            no_layer += 1
            inp_size //= 2

        self.encoder = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(no_layer):
            if i == 0:
                self.encoder.append(DoubleConv(in_channels, start_filter))
            else:
                self.down_samples.append(Down(start_filter // 2, start_filter))
                self.encoder.append(DoubleConv(start_filter, start_filter * 2))
            start_filter *= 2

        # Middle layer
        self.mid = nn.Sequential(
            nn.Conv2d(start_filter // 2, start_filter, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(start_filter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(start_filter, start_filter * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(start_filter * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(no_layer):
            if i == 0:
                self.decoder.append(Up(start_filter * 2, start_filter // 2, bilinear))
            else:
                self.decoder.append(Up(start_filter + start_filter // 2, start_filter // 2, bilinear))
            start_filter //= 2

        # Output layer
        self.outc = nn.Conv2d(start_filter, out_channels, kernel_size=1)

    def forward(self, x):
        x_list = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.down_samples):
                x_list.append(x)
                x = self.down_samples[i](x)

        x = self.mid(x)

        for i in range(len(self.decoder)):
            x = self.decoder[i](x, x_list[-(i + 1)])

        logits = self.outc(x)
        return torch.sigmoid(logits)

# Example usage
height, width, in_channels, num_classes, out_channels = 256, 256, 3, 1, 1
#model = Planet(in_channels, num_classes, height, width)
model = UNet(in_channels, out_channels)
#model = UNetPlusPlus(in_channels, num_classes)
input_tensor = torch.randn(1, in_channels, height, width)
output = model(input_tensor)
print(output.shape) # Should be [1, num_classes, height, width]