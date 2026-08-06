from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class YoloV1Loss(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = pred.shape[0]
        device = pred.device

        pred_cls = pred[..., : self.num_classes]
        pred_boxes = pred[..., self.num_classes :].view(
            batch_size, self.grid_size, self.grid_size, self.boxes_per_cell, 5
        )

        obj_mask = target[..., 0] > 0
        noobj_mask = ~obj_mask

        gt_cls = target[..., 1 : 1 + self.num_classes]
        gt_box = target[..., 1 + self.num_classes : 1 + self.num_classes + 4]

        pred_xy = pred_boxes[..., 1:3]
        pred_wh = pred_boxes[..., 3:5].clamp(min=1.0e-6)
        pred_wh_squared = pred_wh * pred_wh

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size, device=device),
            torch.arange(self.grid_size, device=device),
            indexing="ij",
        )
        grid_x = grid_x.view(1, self.grid_size, self.grid_size, 1).float()
        grid_y = grid_y.view(1, self.grid_size, self.grid_size, 1).float()

        gt_abs_x = (gt_box[..., 0:1] + grid_x) / self.grid_size
        gt_abs_y = (gt_box[..., 1:2] + grid_y) / self.grid_size
        gt_abs_w = gt_box[..., 2:3]
        gt_abs_h = gt_box[..., 3:4]

        pred_abs_x = (pred_xy[..., 0:1] + grid_x.unsqueeze(-1)) / self.grid_size
        pred_abs_y = (pred_xy[..., 1:2] + grid_y.unsqueeze(-1)) / self.grid_size
        pred_abs_w = pred_wh_squared[..., 0:1]
        pred_abs_h = pred_wh_squared[..., 1:2]

        gt_xyxy = self._to_xyxy(torch.cat([gt_abs_x, gt_abs_y, gt_abs_w, gt_abs_h], dim=-1))
        pred_xyxy = self._to_xyxy(torch.cat([pred_abs_x, pred_abs_y, pred_abs_w, pred_abs_h], dim=-1))

        inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0:1])
        inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1:2])
        inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2:3])
        inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3:4])

        inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
        pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0.0) * (
            pred_xyxy[..., 3] - pred_xyxy[..., 1]
        ).clamp(min=0.0)
        gt_area = (gt_xyxy[..., 2:3] - gt_xyxy[..., 0:1]).clamp(min=0.0) * (
            gt_xyxy[..., 3:4] - gt_xyxy[..., 1:2]
        ).clamp(min=0.0)
        union = pred_area + gt_area.squeeze(-1) - inter
        iou = inter / union.clamp(min=1.0e-6)

        best_box = iou.argmax(dim=-1)
        responsible_mask = torch.zeros_like(iou)
        responsible_mask.scatter_(-1, best_box.unsqueeze(-1), 1.0)
        responsible_mask = responsible_mask * obj_mask.unsqueeze(-1).float()

        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)

        gt_xy = gt_box[..., 0:2]
        gt_wh = gt_box[..., 2:4].clamp(min=1.0e-6)

        for box_idx in range(self.boxes_per_cell):
            pred_box = pred_boxes[..., box_idx, :]
            resp = responsible_mask[..., box_idx].unsqueeze(-1)

            coord_loss = coord_loss + self.mse(resp * pred_box[..., 1:3], resp * gt_xy)
            coord_loss = coord_loss + self.mse(resp * pred_box[..., 3:5], resp * torch.sqrt(gt_wh))

            pred_conf = pred_box[..., 0]
            obj_target = iou[..., box_idx] * responsible_mask[..., box_idx]
            obj_loss = obj_loss + self.mse(responsible_mask[..., box_idx] * pred_conf, obj_target)

            noobj_box_mask = noobj_mask.float() + obj_mask.float() * (1.0 - responsible_mask[..., box_idx])
            noobj_loss = noobj_loss + self.mse(noobj_box_mask * pred_conf, torch.zeros_like(pred_conf))

        cls_loss = self.mse(obj_mask.unsqueeze(-1).float() * pred_cls, obj_mask.unsqueeze(-1).float() * gt_cls)

        total = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + cls_loss
        ) / batch_size

        zero = torch.tensor(0.0, device=device)
        return {
            "loss": total,
            "loss_coord": coord_loss / batch_size,
            "loss_obj": obj_loss / batch_size,
            "loss_noobj": noobj_loss / batch_size,
            "loss_cls": cls_loss / batch_size,
            "loss_motion": zero,
        }

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x, y, w, h = boxes.unbind(dim=-1)
        return torch.stack((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0), dim=-1)
