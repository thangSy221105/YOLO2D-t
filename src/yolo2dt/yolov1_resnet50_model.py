from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class YoloV1ResNet50(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 20,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.output_dim = num_classes + boxes_per_cell * 5

        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.pooling = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, grid_size * grid_size * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.classifier(x)
        return x.view(-1, self.grid_size, self.grid_size, self.output_dim)
