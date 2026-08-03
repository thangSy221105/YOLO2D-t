from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Tuple

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        return tuple(
            value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for value in batch
        )
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["target"], batch["motion_mask"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    raise ValueError("Batch must be a dict or tuple/list with image, target, motion_mask")


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return torch.stack((x1, y1, x2, y2), dim=-1)


def box_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)

    x1 = torch.maximum(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    y1 = torch.maximum(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    x2 = torch.minimum(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    y2 = torch.minimum(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])

    inter_w = (x2 - x1).clamp(min=0.0)
    inter_h = (y2 - y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = (boxes1_xyxy[..., 2] - boxes1_xyxy[..., 0]).clamp(min=0.0) * (
        boxes1_xyxy[..., 3] - boxes1_xyxy[..., 1]
    ).clamp(min=0.0)
    area2 = (boxes2_xyxy[..., 2] - boxes2_xyxy[..., 0]).clamp(min=0.0) * (
        boxes2_xyxy[..., 3] - boxes2_xyxy[..., 1]
    ).clamp(min=0.0)

    union = area1 + area2 - inter
    return inter / union.clamp(min=1.0e-6)


def box_diou_loss_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)

    iou = box_iou_xywh(boxes1, boxes2)

    center_dist = (boxes1[..., 0] - boxes2[..., 0]).pow(2) + (boxes1[..., 1] - boxes2[..., 1]).pow(2)

    enclosing_x1 = torch.minimum(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    enclosing_y1 = torch.minimum(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    enclosing_x2 = torch.maximum(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    enclosing_y2 = torch.maximum(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])
    enclosing_diag = (enclosing_x2 - enclosing_x1).pow(2) + (enclosing_y2 - enclosing_y1).pow(2)

    diou = iou - center_dist / enclosing_diag.clamp(min=1.0e-6)
    return 1.0 - diou


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters if p.requires_grad)
