from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MotionFeatureSpec:
    channels: list[int]


class YoloV8MotionAdapter(nn.Module):
    """Wrap a YOLOv8 detect checkpoint and add a lightweight motion head.

    The detector is used as a pretrained feature extractor. We capture the
    multi-scale feature maps that feed the detect head, pool them to a fixed
    7x7 grid, and predict a 4D motion vector per cell.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 448,
        grid_size: int = 7,
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
        self.hidden_dim = hidden_dim
        self.freeze_detector = freeze_detector
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
        self.motion_head = nn.Sequential(
            nn.Conv2d(hidden_dim * len(feature_spec.channels), hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),
        )

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
                qualified = f"{parent_name}.{child_name}" if parent_name else child_name
                yield qualified, parent, child_name, child

    def _upgrade_first_conv_to_six_channels(self) -> nn.Conv2d:
        for _, parent, child_name, child in self._iter_named_modules_with_parent():
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
        motion = self.motion_head(fused)
        return motion.permute(0, 2, 3, 1).contiguous()


class YoloV8MotionLoss(nn.Module):
    def __init__(
        self,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        lambda_motion: float = 1.0,
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.lambda_motion = lambda_motion
        self.loss_type = loss_type

    def forward(
        self,
        pred_motion: torch.Tensor,
        target: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = pred_motion.shape[0]
        target_motion = target[..., self.boxes_per_cell * 5 + self.num_classes :]
        obj_mask = target[..., 4] > 0
        valid = (obj_mask & (motion_mask > 0)).unsqueeze(-1).to(pred_motion.dtype)

        if self.loss_type == "l1":
            raw_motion = F.l1_loss(valid * pred_motion, valid * target_motion, reduction="sum")
        else:
            raw_motion = F.smooth_l1_loss(valid * pred_motion, valid * target_motion, reduction="sum")

        total_loss = (self.lambda_motion * raw_motion) / batch_size
        zero = pred_motion.new_tensor(0.0)
        return {
            "loss": total_loss,
            "loss_coord": zero,
            "loss_obj": zero,
            "loss_noobj": zero,
            "loss_cls": zero,
            "loss_motion": raw_motion / batch_size,
        }
