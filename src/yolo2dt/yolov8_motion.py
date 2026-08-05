from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MotionFeatureSpec:
    channels: list[int]


class YoloV8MotionAdapter(nn.Module):
    """Joint detection + motion head on top of a YOLOv8 detect checkpoint.

    The pretrained YOLOv8 detector is reused as a 6-channel feature extractor.
    We capture the multi-scale tensors that feed the detect head, pool them to
    a fixed 7x7 grid, then predict:

    - detection grid output: B * 5 + C
    - motion output: 4

    The final tensor layout matches the existing YOLOv1-style target:
    ``S x S x (B * 5 + C + 4)``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 448,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        hidden_dim: int = 128,
        freeze_detector: bool = True,
        train_first_conv: bool = True,
    ) -> None:
        super().__init__()

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is required for YoloV8MotionAdapter. Install it with `pip install ultralytics`."
            ) from exc

        self.checkpoint_path = str(checkpoint_path)
        self.image_size = image_size
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.freeze_detector = freeze_detector
        self.motion_dims = 4
        self.detect_dims = boxes_per_cell * 5 + num_classes
        self.output_dim = self.detect_dims + self.motion_dims
        self._detect_inputs: list[torch.Tensor] | None = None

        yolo = YOLO(self.checkpoint_path)
        self.detector = yolo.model
        self._detect_head = self.detector.model[-1]
        self._detect_head.register_forward_pre_hook(self._capture_detect_inputs)

        self._first_conv = self._upgrade_first_conv_to_six_channels()
        if freeze_detector:
            for parameter in self.detector.parameters():
                parameter.requires_grad = False
        if train_first_conv:
            for parameter in self._first_conv.parameters():
                parameter.requires_grad = True

        feature_spec = self._infer_feature_spec()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for channels in feature_spec.channels
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(feature_spec.channels), hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.detect_head = nn.Conv2d(hidden_dim, self.detect_dims, kernel_size=1)
        self.motion_head = nn.Conv2d(hidden_dim, self.motion_dims, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector:
            self.detector.eval()
        return self

    def _capture_detect_inputs(self, module: nn.Module, inputs: tuple[object, ...]) -> None:
        if not inputs:
            self._detect_inputs = None
            return
        raw = inputs[0]
        if isinstance(raw, (list, tuple)):
            self._detect_inputs = [tensor for tensor in raw if torch.is_tensor(tensor)]
        elif torch.is_tensor(raw):
            self._detect_inputs = [raw]
        else:
            self._detect_inputs = None

    def _iter_named_modules_with_parent(self):
        for parent_name, parent in self.detector.named_modules():
            for child_name, child in parent.named_children():
                yield parent, child_name, child

    def _upgrade_first_conv_to_six_channels(self) -> nn.Conv2d:
        for parent, child_name, child in self._iter_named_modules_with_parent():
            if isinstance(child, nn.Conv2d):
                if child.in_channels != 3:
                    raise ValueError(
                        f"Expected the first detector conv to have 3 input channels, found {child.in_channels}."
                    )

                new_conv = nn.Conv2d(
                    in_channels=6,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    new_conv.weight[:, :3].copy_(child.weight * 0.5)
                    new_conv.weight[:, 3:].copy_(child.weight * 0.5)
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(parent, child_name, new_conv)
                return new_conv
        raise RuntimeError("Could not find a Conv2d layer to upgrade to 6-channel input.")

    def _infer_feature_spec(self) -> MotionFeatureSpec:
        was_training = self.detector.training
        self.detector.eval()
        device = next(self.detector.parameters()).device
        dummy = torch.zeros(1, 6, self.image_size, self.image_size, device=device)
        with torch.no_grad():
            _ = self.detector(dummy)
        if self._detect_inputs is None or not self._detect_inputs:
            raise RuntimeError("Failed to capture YOLOv8 detect head input features.")
        channels = [int(feature.shape[1]) for feature in self._detect_inputs]
        self.detector.train(was_training)
        self._detect_inputs = None
        return MotionFeatureSpec(channels=channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._detect_inputs = None
        _ = self.detector(image)
        if self._detect_inputs is None or not self._detect_inputs:
            raise RuntimeError("Detect head inputs were not captured during forward pass.")

        pooled = []
        for feature, projection in zip(self._detect_inputs, self.projections):
            projected = projection(feature)
            pooled.append(F.adaptive_avg_pool2d(projected, (self.grid_size, self.grid_size)))

        fused = torch.cat(pooled, dim=1)
        fused = self.fusion(fused)

        detect = self.detect_head(fused)
        motion = self.motion_head(fused)
        output = torch.cat([detect, motion], dim=1)
        return output.permute(0, 2, 3, 1).contiguous()
