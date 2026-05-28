"""
Convolutional Neural Network for Jet Image Classification
==========================================================
Based on: de Oliveira et al., "Jet-images — deep learning edition" (2016)

Architecture:
  - Input: 40×40×3 jet images (pT, multiplicity, pT² channels)
  - ResNet-style blocks with skip connections
  - Global average pooling → dense classifier

Physics motivation:
  Jet images pixelate the radiation pattern of a jet in the (η,φ) plane,
  analogous to what a calorimeter measures. CNNs learn local patterns
  (subjets, prongs) that distinguish top jets from QCD jets.

  Top quark jets show 3-prong structure (from t → Wb → qqb),
  while QCD jets are typically 1-prong with diffuse radiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_CONFIG, IMG_SIZE, IMG_CHANNELS


class ResBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: match dimensions if needed
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)  # Skip connection
        return F.relu(out)


class JetCNN(nn.Module):
    """
    CNN classifier for jet images.

    Architecture:
      Conv(3→64) → ResBlock(64→64) → Pool
      → ResBlock(64→128) → Pool
      → ResBlock(128→256) → Pool
      → GlobalAvgPool → FC(256→128) → FC(128→1)
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or CNN_CONFIG
        filters = cfg["filters"]
        dropout = cfg["dropout"]

        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, filters[0], 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with progressive downsampling
        self.block1 = ResBlock(filters[0], filters[0])
        self.pool1 = nn.MaxPool2d(2)  # 40→20

        self.block2 = ResBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)  # 20→10

        self.block3 = ResBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)  # 10→5

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters[2], cfg["fc_dim"]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cfg["fc_dim"], 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 40, 40) jet images

        Returns:
            logits: (batch, 1) raw logits (apply sigmoid for probability)
        """
        x = self.conv_in(x)

        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch, 256)
        return self.classifier(x).squeeze(-1)  # (batch,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = JetCNN()
    print(f"JetCNN: {model.count_parameters():,} trainable parameters")
    x = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
