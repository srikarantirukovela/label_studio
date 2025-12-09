import torch
import torch.nn as nn

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from tqdm import tqdm
# from torchsummary import summary
import albumentations as A
# from torchinfo import summary
from albumentations.pytorch import ToTensorV2
import warnings
# import numpy as np
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== RESUNET-A ARCHITECTURE ====================
class ResidualConvBlock(nn.Module):
    """Residual Convolutional Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels=256):
        super(ASPPModule, self).__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=6
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=12
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=18
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Branch 1: 1x1 convolution
        branch1 = self.conv1x1(x)
        
        # Branch 2: 3x3 dilation=6
        branch2 = self.conv3x3_1(x)
        
        # Branch 3: 3x3 dilation=12
        branch3 = self.conv3x3_2(x)
        
        # Branch 4: 3x3 dilation=18
        branch4 = self.conv3x3_3(x)
        
        # Branch 5: Global average pooling
        branch5 = self.global_avg_pool(x)
        branch5 = F.interpolate(branch5, size=(h, w), mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        out = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        out = self.conv1x1_out(out)
        
        return out

class AttentionGate(nn.Module):
    """Attention Gate for better feature fusion"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            # print(f"Resizing g1 from {g1.shape[2:]} to {x1.shape[2:]}")
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResUNetA(nn.Module):
    """ResUNet-a: Advanced UNet with Residual blocks, ASPP, and Attention"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(ResUNetA, self).__init__()
        
        # Encoder path
        self.encoder1 = ResidualConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ResidualConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ResidualConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = ResidualConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge with ASPP
        self.bridge = ResidualConvBlock(features[3], features[4])
        self.aspp = ASPPModule(features[4], features[4] // 2)
        
        # Decoder path with attention gates
        self.attention4 = AttentionGate(F_g=features[4] // 2, F_l=features[3], F_int=features[3] // 2)
        self.up_conv4 = nn.ConvTranspose2d(features[4] // 2, features[3], kernel_size=2, stride=2)
        self.decoder4 = ResidualConvBlock(features[3] * 2, features[3])
        
        self.attention3 = AttentionGate(F_g=features[3], F_l=features[2], F_int=features[2] // 2)
        self.up_conv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ResidualConvBlock(features[2] * 2, features[2])
        
        self.attention2 = AttentionGate(F_g=features[2], F_l=features[1], F_int=features[1] // 2)
        self.up_conv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ResidualConvBlock(features[1] * 2, features[1])
        
        self.attention1 = AttentionGate(F_g=features[1], F_l=features[0], F_int=features[0] // 2)
        self.up_conv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ResidualConvBlock(features[0] * 2, features[0])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # Use sigmoid for binary segmentation
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(self.pool1(e1))  # 128 channels
        e3 = self.encoder3(self.pool2(e2))  # 256 channels
        e4 = self.encoder4(self.pool3(e3))  # 512 channels
        
        # Bridge with ASPP
        bridge = self.bridge(self.pool4(e4))  # 1024 channels
        bridge = self.aspp(bridge)  # 512 channels
        
        # Decoder with attention
        # d4 = self.up_conv4(bridge)
        # a4 = self.attention4(g=bridge, x=e4)
        a4 = self.attention4(g=bridge, x=e4)
        d4 = self.up_conv4(bridge)
        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.up_conv3(d4)
        a3 = self.attention3(g=d4, x=e3)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.up_conv2(d3)
        a2 = self.attention2(g=d3, x=e2)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.up_conv1(d2)
        a1 = self.attention1(g=d2, x=e1)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        # Final output
        out = self.final_conv(d1)
        return out
