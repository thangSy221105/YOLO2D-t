from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.mot17_2dt_dataset import box_xywh_to_xyxy, decode_target_boxes, draw_motion_arrow, tensor_to_pil
from yolo2dt.config import load_config
from yolo2dt.data_adapter import build_dataloaders
from yolo2dt.yolov8_model import YoloV8Style2DT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolov8_2dt.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="outputs/yolov8_2dt/prediction_preview.jpg")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def latest_checkpoint(output_dir: str | Path) -> Path:
    checkpoints = sorted(Path(output_dir).glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {output_dir}")
    return checkpoints[-1]


def decode_prediction_boxes(
    pred: torch.Tensor,
    image_size: int,
    grid_size: int,
    boxes_per_cell: int,
    num_classes: int,
    conf_thresh: float,
) -> list[dict[str, float]]:
    pred = pred.detach().cpu()
    class_offset = boxes_per_cell * 5
    motion_offset = class_offset + num_classes
    boxes: list[dict[str, float]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            best = None
            best_conf = -1.0
            for box_idx in range(boxes_per_cell):
                offset = box_idx * 5
                conf = float(pred[row, col, offset + 4])
                if conf > best_conf:
                    best_conf = conf
                    best = (
                        float(pred[row, col, offset + 0]),
                        float(pred[row, col, offset + 1]),
                        float(pred[row, col, offset + 2]),
                        float(pred[row, col, offset + 3]),
                        conf,
                    )

            if best is None or best_conf < conf_thresh:
                continue

            x_cell, y_cell, w_norm, h_norm, conf = best
            x_cell = min(max(x_cell, 0.0), 1.0)
            y_cell = min(max(y_cell, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            x = ((col + x_cell) / grid_size) * image_size
            y = ((row + y_cell) / grid_size) * image_size
            w = w_norm * image_size
            h = h_norm * image_size

            mx = float(pred[row, col, motion_offset + 0]) * image_size
            my = float(pred[row, col, motion_offset + 1]) * image_size
            mw = float(pred[row, col, motion_offset + 2]) * image_size
            mh = float(pred[row, col, motion_offset + 3]) * image_size

            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "end_x": x + mx,
                    "end_y": y + my,
                    "end_w": w + mw,
                    "end_h": h + mh,
                    "conf": conf,
                    "cell_row": float(row),
                    "cell_col": float(col),
                }
            )

    return boxes


def draw_boxes(panel: Image.Image, boxes: list[dict[str, float]], color: tuple[int, int, int], show_conf: bool) -> None:
    draw = ImageDraw.Draw(panel)
    for obj in boxes:
        box = box_xywh_to_xyxy(obj["x"], obj["y"], obj["w"], obj["h"])
        draw.rectangle(box, outline=color, width=2)
        draw_motion_arrow(draw, obj["x"], obj["y"], obj["end_x"], obj["end_y"], fill=color)
        if show_conf:
            draw.text((box[0], max(0, box[1] - 12)), f"{obj['conf']:.2f}", fill=color)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    requested_device = cfg["train"].get("device", "cuda")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    _, val_loader = build_dataloaders(cfg)
    batch = next(iter(val_loader))
    if isinstance(batch, dict):
        images = batch["image"]
        targets = batch["target"]
        motion_masks = batch["motion_mask"]
    else:
        images, targets, motion_masks = batch

    sample_idx = min(max(args.sample, 0), images.shape[0] - 1)

    model = YoloV8Style2DT(
        in_channels=cfg["model"]["in_channels"],
        grid_size=cfg["data"]["grid_size"],
        boxes_per_cell=cfg["data"]["boxes_per_cell"],
        num_classes=cfg["data"]["num_classes"],
        width=cfg["model"].get("width", 0.5),
        depth=cfg["model"].get("depth", 0.33),
        dropout=cfg["model"].get("dropout", 0.1),
        activate_output=cfg["model"].get("activate_output", True),
    ).to(device)

    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(cfg["train"]["output_dir"])
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(images.to(device)).cpu()

    image_size = cfg["data"]["image_size"]
    frame = tensor_to_pil(images[sample_idx, :3])

    gt_panel = frame.copy()
    pred_panel = frame.copy()

    gt_boxes = decode_target_boxes(
        targets[sample_idx],
        motion_masks[sample_idx],
        image_size=image_size,
        grid_size=cfg["data"]["grid_size"],
        boxes_per_cell=cfg["data"]["boxes_per_cell"],
        num_classes=cfg["data"]["num_classes"],
    )
    pred_boxes = decode_prediction_boxes(
        preds[sample_idx],
        image_size=image_size,
        grid_size=cfg["data"]["grid_size"],
        boxes_per_cell=cfg["data"]["boxes_per_cell"],
        num_classes=cfg["data"]["num_classes"],
        conf_thresh=args.conf,
    )

    draw_boxes(gt_panel, gt_boxes, color=(60, 220, 110), show_conf=False)
    draw_boxes(pred_panel, pred_boxes, color=(255, 80, 80), show_conf=True)

    canvas = Image.new("RGB", (image_size * 2, image_size), (30, 30, 30))
    canvas.paste(gt_panel, (0, 0))
    canvas.paste(pred_panel, (image_size, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"Ground truth ({len(gt_boxes)})", fill=(60, 220, 110))
    draw.text((image_size + 8, 8), f"Prediction ({len(pred_boxes)}) | {checkpoint.name}", fill=(255, 80, 80))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved visualization: {output}")


if __name__ == "__main__":
    main()
