from __future__ import annotations

import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.cv1 = Conv(channels, hidden, kernel_size=1)
        self.cv2 = Conv(hidden, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C2f(nn.Module):
    """Compact YOLOv8-style C2f block."""

    def __init__(self, in_channels: int, out_channels: int, n: int = 1) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 1)
        self.cv1 = Conv(in_channels, hidden * 2, kernel_size=1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, shortcut=True) for _ in range(n))
        self.cv2 = Conv(hidden * (2 + n), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = max(in_channels // 2, 1)
        self.cv1 = Conv(in_channels, hidden, kernel_size=1)
        self.cv2 = Conv(hidden * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class YoloV8Style2DT(nn.Module):
    """YOLOv8-style 2D+t prototype with 6-channel input and a motion-aware grid head.

    The head intentionally emits the same target layout as the YOLOv1-style
    scaffold: S x S x (B*5 + C + 4). This keeps the experiment compatible with
    the current dataset and loss while changing the backbone/head style.
    """

    def __init__(
        self,
        in_channels: int = 6,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        width: float = 0.5,
        depth: float = 0.33,
        dropout: float = 0.1,
        activate_output: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.motion_dims = 4
        self.output_dim = boxes_per_cell * 5 + num_classes + self.motion_dims
        self.activate_output = activate_output

        def c(channels: int) -> int:
            return max(int(channels * width), 8)

        def d(repeats: int) -> int:
            return max(round(repeats * depth), 1)

        self.backbone = nn.Sequential(
            Conv(in_channels, c(64), 3, 2),
            Conv(c(64), c(128), 3, 2),
            C2f(c(128), c(128), d(3)),
            Conv(c(128), c(256), 3, 2),
            C2f(c(256), c(256), d(6)),
            Conv(c(256), c(512), 3, 2),
            C2f(c(512), c(512), d(6)),
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), d(3)),
            SPPF(c(1024), c(1024)),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

        head_channels = c(512)
        self.head = nn.Sequential(
            Conv(c(1024), head_channels, 3, 1),
            nn.Dropout(dropout),
            nn.Conv2d(head_channels, self.output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.head(self.backbone(x)).permute(0, 2, 3, 1).contiguous()
        if not self.activate_output:
            return pred
        return self._activate(pred)

    def _activate(self, pred: torch.Tensor) -> torch.Tensor:
        chunks = []
        for box_idx in range(self.boxes_per_cell):
            start = box_idx * 5
            xywh_conf = torch.sigmoid(pred[..., start : start + 5])
            chunks.append(xywh_conf)

        class_start = self.boxes_per_cell * 5
        class_end = class_start + self.num_classes
        chunks.append(torch.sigmoid(pred[..., class_start:class_end]))
        chunks.append(torch.tanh(pred[..., class_end : class_end + self.motion_dims]))
        return torch.cat(chunks, dim=-1)
