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
    """Frozen YOLOv8 detector features plus auxiliary motion/future heads.

    Detection remains handled by the original YOLOv8 checkpoint. This module
    learns a 7x7x4 motion map and a 7x7x4 future-box map from two-frame input,
    so motion training can use future supervision without overwriting the
    detector head weights used for detection metrics.
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
        self.future_head = nn.Sequential(
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
        for _, parent in self.detector.named_modules():
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

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        future = self.future_head(fused)
        return {
            "motion": motion.permute(0, 2, 3, 1).contiguous(),
            "future": future.permute(0, 2, 3, 1).contiguous(),
        }


class YoloV8MotionLoss(nn.Module):
    def __init__(
        self,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        lambda_motion: float = 1.0,
        loss_type: str = "smooth_l1",
        lambda_direction: float = 1.0,
        lambda_future: float = 1.0,
        future_loss_type: str = "smooth_l1",
        moving_threshold: float = 1.0e-2,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.lambda_motion = lambda_motion
        self.loss_type = loss_type
        self.lambda_direction = lambda_direction
        self.lambda_future = lambda_future
        self.future_loss_type = future_loss_type
        self.moving_threshold = moving_threshold
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor | Dict[str, torch.Tensor],
        target: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(predictions, dict):
            pred_motion = predictions["motion"]
            pred_future_raw = predictions.get("future")
        else:
            pred_motion = predictions
            pred_future_raw = None

        batch_size = pred_motion.shape[0]
        target_motion = target[..., self.boxes_per_cell * 5 + self.num_classes :]
        obj_mask = target[..., 4] > 0
        valid_mask = obj_mask & (motion_mask > 0)
        valid = valid_mask.unsqueeze(-1).to(pred_motion.dtype)

        if self.loss_type == "l1":
            raw_motion = F.l1_loss(valid * pred_motion, valid * target_motion, reduction="sum")
        else:
            raw_motion = F.smooth_l1_loss(valid * pred_motion, valid * target_motion, reduction="sum")

        # Separate direction supervision helps when magnitude is learned but heading is unstable.
        target_xy = target_motion[..., :2]
        pred_xy = pred_motion[..., :2]
        target_speed = torch.sqrt((target_xy**2).sum(dim=-1) + self.eps)
        moving_mask = valid_mask & (target_speed > self.moving_threshold)

        if moving_mask.any():
            pred_xy_m = pred_xy[moving_mask]
            target_xy_m = target_xy[moving_mask]

            pred_xy_unit = pred_xy_m / (torch.norm(pred_xy_m, dim=-1, keepdim=True) + self.eps)
            target_xy_unit = target_xy_m / (torch.norm(target_xy_m, dim=-1, keepdim=True) + self.eps)

            cosine = (pred_xy_unit * target_xy_unit).sum(dim=-1)
            raw_direction = (1.0 - cosine).sum()
            mean_cosine = cosine.mean()
        else:
            raw_direction = pred_motion.new_tensor(0.0)
            mean_cosine = pred_motion.new_tensor(0.0)

        if pred_future_raw is not None:
            grid_size = pred_motion.shape[1]
            row_index = torch.arange(grid_size, device=target.device, dtype=target.dtype).view(1, grid_size, 1)
            col_index = torch.arange(grid_size, device=target.device, dtype=target.dtype).view(1, 1, grid_size)

            current_x = (target[..., 0] + col_index) / grid_size
            current_y = (target[..., 1] + row_index) / grid_size
            current_w = target[..., 2]
            current_h = target[..., 3]

            target_future = torch.stack(
                [
                    current_x + target_motion[..., 0],
                    current_y + target_motion[..., 1],
                    torch.clamp(current_w + target_motion[..., 2], min=self.eps),
                    torch.clamp(current_h + target_motion[..., 3], min=self.eps),
                ],
                dim=-1,
            )
            pred_future = torch.sigmoid(pred_future_raw)

            if self.future_loss_type == "l1":
                raw_future = F.l1_loss(valid * pred_future, valid * target_future, reduction="sum")
            else:
                raw_future = F.smooth_l1_loss(valid * pred_future, valid * target_future, reduction="sum")
        else:
            raw_future = pred_motion.new_tensor(0.0)

        total_loss = (
            self.lambda_motion * raw_motion
            + self.lambda_direction * raw_direction
            + self.lambda_future * raw_future
        ) / batch_size
        zero = pred_motion.new_tensor(0.0)
        return {
            "loss": total_loss,
            "loss_coord": zero,
            "loss_obj": zero,
            "loss_noobj": zero,
            "loss_cls": zero,
            "loss_motion": raw_motion / batch_size,
            "loss_direction": raw_direction / batch_size,
            "loss_future": raw_future / batch_size,
            "mean_cosine": mean_cosine,
            "moving_cells": moving_mask.sum().to(pred_motion.dtype),
        }
