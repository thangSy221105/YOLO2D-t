from __future__ import annotations

import torch
from torch import nn


def _conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YoloV1Original(nn.Module):
    def __init__(self, grid_size: int = 7, boxes_per_cell: int = 2, num_classes: int = 20) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.output_dim = num_classes + boxes_per_cell * 5

        layers: list[nn.Module] = [
            _conv(3, 64, 7, stride=2),
            nn.MaxPool2d(2, 2),
            _conv(64, 192, 3),
            nn.MaxPool2d(2, 2),
            _conv(192, 128, 1),
            _conv(128, 256, 3),
            _conv(256, 256, 1),
            _conv(256, 512, 3),
            nn.MaxPool2d(2, 2),
        ]

        for _ in range(4):
            layers.extend([_conv(512, 256, 1), _conv(256, 512, 3)])

        layers.extend(
            [
                _conv(512, 512, 1),
                _conv(512, 1024, 3),
                nn.MaxPool2d(2, 2),
                _conv(1024, 512, 1),
                _conv(512, 1024, 3),
                _conv(1024, 512, 1),
                _conv(512, 1024, 3),
                _conv(1024, 1024, 3),
                _conv(1024, 1024, 3, stride=2),
                _conv(1024, 1024, 3),
                _conv(1024, 1024, 3),
            ]
        )

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.view(-1, self.grid_size, self.grid_size, self.output_dim)
