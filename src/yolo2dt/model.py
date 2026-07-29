from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, kernel_size: int, stride: int) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Yolo2DTiny(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.motion_dims = 4
        self.output_dim = boxes_per_cell * 5 + num_classes + self.motion_dims

        self.backbone = nn.Sequential(
            conv_block(in_channels, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            conv_block(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256, 3, 1),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512, 3, 1),
            nn.MaxPool2d(2, 2),
            conv_block(512, 1024, 3, 1),
            conv_block(1024, 1024, 3, 1),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

        self.head = nn.Sequential(
            nn.Conv2d(1024, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, self.output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pred = self.head(features)
        return pred.permute(0, 2, 3, 1).contiguous()
