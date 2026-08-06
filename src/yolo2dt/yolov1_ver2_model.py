from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class YoloV1Ver2(nn.Module):
    """Checkpoint-compatible YOLOv1 variant.

    This matches the observed checkpoint structure:
    - `darknet.<idx>.*` for the convolutional backbone
    - `fcs.1` and `fcs.4` for the linear head
    """

    def __init__(self, grid_size: int = 7, boxes_per_cell: int = 2, num_classes: int = 20) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.output_dim = num_classes + boxes_per_cell * 5

        self.darknet = nn.Sequential(
            ConvBNAct(3, 64, 7, stride=2),  # 0
            nn.MaxPool2d(2, 2),             # 1
            ConvBNAct(64, 192, 3),          # 2
            nn.MaxPool2d(2, 2),             # 3
            ConvBNAct(192, 128, 1),         # 4
            ConvBNAct(128, 256, 3),         # 5
            ConvBNAct(256, 256, 1),         # 6
            ConvBNAct(256, 512, 3),         # 7
            nn.MaxPool2d(2, 2),             # 8
            ConvBNAct(512, 256, 1),         # 9
            ConvBNAct(256, 512, 3),         # 10
            ConvBNAct(512, 256, 1),         # 11
            ConvBNAct(256, 512, 3),         # 12
            ConvBNAct(512, 256, 1),         # 13
            ConvBNAct(256, 512, 3),         # 14
            ConvBNAct(512, 256, 1),         # 15
            ConvBNAct(256, 512, 3),         # 16
            ConvBNAct(512, 512, 1),         # 17
            ConvBNAct(512, 1024, 3),        # 18
            nn.MaxPool2d(2, 2),             # 19
            ConvBNAct(1024, 512, 1),        # 20
            ConvBNAct(512, 1024, 3),        # 21
            ConvBNAct(1024, 512, 1),        # 22
            ConvBNAct(512, 1024, 3),        # 23
            ConvBNAct(1024, 1024, 3),       # 24
            ConvBNAct(1024, 1024, 3, stride=2),  # 25
            ConvBNAct(1024, 1024, 3),       # 26
            ConvBNAct(1024, 1024, 3),       # 27
        )

        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 496),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(496, grid_size * grid_size * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.darknet(x)
        x = self.fcs(x)
        return x.view(-1, self.grid_size, self.grid_size, self.output_dim)
