from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .utils import box_iou_xywh


@dataclass
class LossBreakdown:
    total: torch.Tensor
    coord: torch.Tensor
    obj: torch.Tensor
    noobj: torch.Tensor
    cls: torch.Tensor
    motion: torch.Tensor


class Yolo2DTLoss(nn.Module):
    def __init__(
        self,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0,
        lambda_motion: float = 1.0,
    ) -> None:
        super().__init__()
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_motion = lambda_motion
        self.mse = nn.MSELoss(reduction="sum")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = pred.shape[0]
        device = pred.device

        pred_boxes = pred[..., : self.boxes_per_cell * 5].view(
            batch_size, pred.shape[1], pred.shape[2], self.boxes_per_cell, 5
        )
        pred_cls = pred[..., self.boxes_per_cell * 5 : self.boxes_per_cell * 5 + self.num_classes]
        pred_motion = pred[..., self.boxes_per_cell * 5 + self.num_classes :]

        target_boxes = target[..., : self.boxes_per_cell * 5].view(
            batch_size, target.shape[1], target.shape[2], self.boxes_per_cell, 5
        )
        target_cls = target[..., self.boxes_per_cell * 5 : self.boxes_per_cell * 5 + self.num_classes]
        target_motion = target[..., self.boxes_per_cell * 5 + self.num_classes :]

        gt_box = target_boxes[..., 0, :4]
        obj_mask = target_boxes[..., 0, 4] > 0
        noobj_mask = ~obj_mask

        ious = []
        for box_idx in range(self.boxes_per_cell):
            iou = box_iou_xywh(pred_boxes[..., box_idx, :4], gt_box)
            ious.append(iou.unsqueeze(-1))
        ious = torch.cat(ious, dim=-1)
        best_box_idx = ious.argmax(dim=-1)

        responsible_mask = torch.zeros(
            batch_size, pred.shape[1], pred.shape[2], self.boxes_per_cell, device=device, dtype=pred.dtype
        )
        responsible_mask.scatter_(-1, best_box_idx.unsqueeze(-1), 1.0)
        responsible_mask = responsible_mask * obj_mask.unsqueeze(-1).to(pred.dtype)

        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)

        gt_xy = gt_box[..., :2]
        gt_wh = gt_box[..., 2:4].clamp(min=1.0e-6)

        for box_idx in range(self.boxes_per_cell):
            box_pred = pred_boxes[..., box_idx, :]
            box_resp = responsible_mask[..., box_idx].unsqueeze(-1)

            pred_xy = box_pred[..., :2]
            pred_wh = box_pred[..., 2:4].clamp(min=1.0e-6)
            pred_conf = box_pred[..., 4:5]

            coord_loss = coord_loss + self.mse(box_resp * pred_xy, box_resp * gt_xy)
            coord_loss = coord_loss + self.mse(box_resp * torch.sqrt(pred_wh), box_resp * torch.sqrt(gt_wh))

            target_conf = box_resp
            obj_loss = obj_loss + self.mse(box_resp * pred_conf, target_conf)

            noobj_box_mask = noobj_mask.unsqueeze(-1).to(pred.dtype) + (
                obj_mask.unsqueeze(-1).to(pred.dtype) * (1.0 - box_resp)
            )
            noobj_loss = noobj_loss + self.mse(noobj_box_mask * pred_conf, torch.zeros_like(pred_conf))

        cls_mask = obj_mask.unsqueeze(-1).to(pred.dtype)
        cls_loss = self.mse(cls_mask * pred_cls, cls_mask * target_cls)

        motion_valid_mask = (obj_mask & (motion_mask > 0)).unsqueeze(-1).to(pred.dtype)
        motion_loss = self.mse(motion_valid_mask * pred_motion, motion_valid_mask * target_motion)

        total_loss = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * cls_loss
            + self.lambda_motion * motion_loss
        ) / batch_size

        return {
            "loss": total_loss,
            "loss_coord": coord_loss / batch_size,
            "loss_obj": obj_loss / batch_size,
            "loss_noobj": noobj_loss / batch_size,
            "loss_cls": cls_loss / batch_size,
            "loss_motion": motion_loss / batch_size,
        }
