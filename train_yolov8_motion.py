from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo2dt.config import load_config
from yolo2dt.data_adapter import build_dataloaders
from yolo2dt.trainer import run_epoch, save_checkpoint, save_history
from yolo2dt.utils import count_parameters, ensure_dir, set_seed
from yolo2dt.yolov8_motion import YoloV8MotionAdapter, YoloV8MotionLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolov8_motion.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="YOLOv8 detect checkpoint (.pt or model name).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-motion", type=float, default=None)
    parser.add_argument("--lambda-direction", type=float, default=None)
    parser.add_argument("--lambda-future", type=float, default=None)
    parser.add_argument("--loss-type", type=str, default="smooth_l1", choices=["smooth_l1", "l1"])
    parser.add_argument("--future-loss-type", type=str, default=None, choices=["smooth_l1", "l1"])
    parser.add_argument("--moving-threshold", type=float, default=None)
    parser.add_argument("--unfreeze-detector", action="store_true", help="Train YOLOv8 feature extractor too.")
    parser.add_argument("--freeze-first-conv", action="store_true", help="Keep upgraded 6-channel stem frozen.")
    parser.add_argument("--patience", type=int, default=5, help="Early stop based on validation motion loss.")
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--benchmark-every", type=int, default=1)
    parser.add_argument("--detect-data", type=str, default=None, help="YOLO detection data.yaml for source detector metrics.")
    parser.add_argument("--detect-imgsz", type=int, default=640)
    parser.add_argument("--detect-batch", type=int, default=32)
    return parser.parse_args()


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_best_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict[str, list],
    best_val_motion: float,
    source_checkpoint: str,
) -> Path:
    output_dir = ensure_dir(output_dir)
    checkpoint_path = output_dir / "best_motion.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_motion": best_val_motion,
            "source_checkpoint": source_checkpoint,
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_source_detector_benchmark(
    checkpoint: str,
    data_yaml: str,
    imgsz: int,
    batch: int,
    device: torch.device,
) -> dict[str, float]:
    from ultralytics import YOLO

    yolo = YOLO(checkpoint)
    device_arg = 0 if device.type == "cuda" else "cpu"
    metrics = yolo.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device_arg,
        verbose=False,
    )
    return {
        "detect_precision": float(metrics.box.mp),
        "detect_recall": float(metrics.box.mr),
        "detect_map50": float(metrics.box.map50),
        "detect_map50_95": float(metrics.box.map),
    }


def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    return xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0


def box_iou_single(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
    area1 = max(x12 - x11, 0.0) * max(y12 - y11, 0.0)
    area2 = max(x22 - x21, 0.0) * max(y22 - y21, 0.0)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def benchmark_motion_val(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    from yolo2dt.utils import move_batch_to_device, unpack_batch

    model.eval()
    grid_size = int(config["data"]["grid_size"])
    boxes_per_cell = int(config["data"]["boxes_per_cell"])
    num_classes = int(config["data"]["num_classes"])
    motion_offset = boxes_per_cell * 5 + num_classes

    valid_count = 0
    l1_sum = torch.zeros(4, dtype=torch.float64)
    center_l2_sum = 0.0
    future_iou_sum = 0.0
    future_head_center_l2_sum = 0.0
    future_head_iou_sum = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images, targets, motion_masks = unpack_batch(batch)
        predictions = model(images)
        if isinstance(predictions, dict):
            preds_motion = predictions["motion"]
            preds_future_raw = predictions.get("future")
        else:
            preds_motion = predictions
            preds_future_raw = None
        gt_motion = targets[..., motion_offset : motion_offset + 4]
        obj_mask = targets[..., 4] > 0
        valid_mask = obj_mask & (motion_masks > 0)

        if valid_mask.sum() == 0:
            continue

        pred_valid = preds_motion[valid_mask]
        gt_valid = gt_motion[valid_mask]
        valid_count += int(valid_mask.sum().item())
        l1_sum += (pred_valid - gt_valid).abs().sum(dim=0).detach().cpu().double()
        center_l2_sum += float(torch.sqrt(((pred_valid[:, :2] - gt_valid[:, :2]) ** 2).sum(dim=1)).sum().item())

        valid_indices = valid_mask.nonzero(as_tuple=False)
        for index in valid_indices:
            batch_idx, row, col = [int(value.item()) for value in index]
            x_cell = float(targets[batch_idx, row, col, 0].item())
            y_cell = float(targets[batch_idx, row, col, 1].item())
            w = float(targets[batch_idx, row, col, 2].item())
            h = float(targets[batch_idx, row, col, 3].item())
            xc = (col + x_cell) / grid_size
            yc = (row + y_cell) / grid_size

            gt = gt_motion[batch_idx, row, col].detach().cpu()
            pred = preds_motion[batch_idx, row, col].detach().cpu()
            if preds_future_raw is not None:
                pred_future_abs = torch.sigmoid(preds_future_raw[batch_idx, row, col]).detach().cpu()
            else:
                pred_future_abs = None
            gt_future = xywh_to_xyxy(
                xc + float(gt[0]),
                yc + float(gt[1]),
                max(w + float(gt[2]), 1.0e-6),
                max(h + float(gt[3]), 1.0e-6),
            )
            pred_future = xywh_to_xyxy(
                xc + float(pred[0]),
                yc + float(pred[1]),
                max(w + float(pred[2]), 1.0e-6),
                max(h + float(pred[3]), 1.0e-6),
            )
            future_iou_sum += box_iou_single(gt_future, pred_future)
            if pred_future_abs is not None:
                pred_future_head = xywh_to_xyxy(
                    float(pred_future_abs[0]),
                    float(pred_future_abs[1]),
                    max(float(pred_future_abs[2]), 1.0e-6),
                    max(float(pred_future_abs[3]), 1.0e-6),
                )
                future_head_iou_sum += box_iou_single(gt_future, pred_future_head)
                future_head_center_l2_sum += (
                    (float(pred_future_abs[0]) - (xc + float(gt[0]))) ** 2
                    + (float(pred_future_abs[1]) - (yc + float(gt[1]))) ** 2
                ) ** 0.5

    mean_l1 = l1_sum / max(valid_count, 1)
    metrics = {
        "motion_valid_cells": float(valid_count),
        "motion_l1_mx": float(mean_l1[0]),
        "motion_l1_my": float(mean_l1[1]),
        "motion_l1_mw": float(mean_l1[2]),
        "motion_l1_mh": float(mean_l1[3]),
        "motion_center_l2": center_l2_sum / max(valid_count, 1),
        "motion_future_iou": future_iou_sum / max(valid_count, 1),
    }
    if future_head_iou_sum > 0.0 or future_head_center_l2_sum > 0.0:
        metrics["future_head_center_l2"] = future_head_center_l2_sum / max(valid_count, 1)
        metrics["future_head_iou"] = future_head_iou_sum / max(valid_count, 1)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = ensure_dir(args.output_dir or config["train"]["output_dir"])

    requested_device = config["train"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    train_loader, val_loader = build_dataloaders(config)

    model = YoloV8MotionAdapter(
        checkpoint_path=args.checkpoint,
        image_size=config["data"]["image_size"],
        grid_size=config["data"]["grid_size"],
        hidden_dim=config["model"]["hidden_dim"],
        freeze_detector=not args.unfreeze_detector,
        train_first_conv=not args.freeze_first_conv,
    ).to(device)

    criterion = YoloV8MotionLoss(
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        lambda_motion=float(args.lambda_motion if args.lambda_motion is not None else config["loss"]["lambda_motion"]),
        loss_type=args.loss_type,
        lambda_direction=float(
            args.lambda_direction if args.lambda_direction is not None else config["loss"].get("lambda_direction", 1.0)
        ),
        lambda_future=float(
            args.lambda_future if args.lambda_future is not None else config["loss"].get("lambda_future", 1.0)
        ),
        future_loss_type=args.future_loss_type or config["loss"].get("future_loss_type", args.loss_type),
        moving_threshold=float(
            args.moving_threshold if args.moving_threshold is not None else config["loss"].get("moving_threshold", 1.0e-2)
        ),
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr if args.lr is not None else config["train"]["lr"]),
        weight_decay=config["train"]["weight_decay"],
    )

    total_epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    history = {"train": [], "val": []}
    best_val_key = "loss_future" if criterion.lambda_future > 0 else "loss_motion"
    best_val_metric = float("inf")
    bad_epochs = 0

    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    print("model: YOLOv8 detector + auxiliary motion/future heads")
    print(f"device: {device}")
    print(f"source checkpoint: {args.checkpoint}")
    print(f"fine-tune epochs: {total_epochs}")
    print(f"output dir: {output_dir}")
    print(f"current lr: {current_lr(optimizer):.3e}")
    print(f"freeze detector: {not args.unfreeze_detector}")
    print(f"train first conv: {not args.freeze_first_conv}")
    print(f"loss type: {args.loss_type}")
    print(f"lambda motion: {criterion.lambda_motion:.4f}")
    print(f"lambda direction: {criterion.lambda_direction:.4f}")
    print(f"lambda future: {criterion.lambda_future:.4f}")
    print(f"future loss type: {criterion.future_loss_type}")
    print(f"moving threshold: {criterion.moving_threshold:.4e}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"trainable params: {count_parameters(model.parameters()):,}")

    if args.detect_data:
        detect_metrics = run_source_detector_benchmark(
            checkpoint=args.checkpoint,
            data_yaml=args.detect_data,
            imgsz=args.detect_imgsz,
            batch=args.detect_batch,
            device=device,
        )
        history.setdefault("source_detector", []).append(detect_metrics)
        print(
            "source detector benchmark:"
            f" P={detect_metrics['detect_precision']:.4f}"
            f" R={detect_metrics['detect_recall']:.4f}"
            f" mAP50={detect_metrics['detect_map50']:.4f}"
            f" mAP50-95={detect_metrics['detect_map50_95']:.4f}"
        )

    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            mixed_precision=mixed_precision,
            grad_clip_norm=config["train"].get("grad_clip_norm"),
            log_interval=config["train"].get("log_interval", 10),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            train=False,
            mixed_precision=mixed_precision,
            grad_clip_norm=None,
            log_interval=config["train"].get("log_interval", 10),
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"train loss: {train_metrics['loss']:.4f} | val loss: {val_metrics['loss']:.4f}")
        print(f"train motion: {train_metrics['loss_motion']:.4f}")
        print(f"val motion: {val_metrics['loss_motion']:.4f}")
        if "loss_future" in train_metrics or "loss_future" in val_metrics:
            print(
                f"train future: {float(train_metrics.get('loss_future', 0.0)):.4f}"
                f" | val future: {float(val_metrics.get('loss_future', 0.0)):.4f}"
            )
        if "loss_direction" in train_metrics or "loss_direction" in val_metrics:
            print(
                f"train direction: {float(train_metrics.get('loss_direction', 0.0)):.4f}"
                f" | val direction: {float(val_metrics.get('loss_direction', 0.0)):.4f}"
            )
        if "mean_cosine" in train_metrics or "mean_cosine" in val_metrics:
            print(
                f"train mean cosine: {float(train_metrics.get('mean_cosine', 0.0)):.4f}"
                f" | val mean cosine: {float(val_metrics.get('mean_cosine', 0.0)):.4f}"
            )
        if "moving_cells" in train_metrics or "moving_cells" in val_metrics:
            print(
                f"train moving cells: {float(train_metrics.get('moving_cells', 0.0)):.1f}"
                f" | val moving cells: {float(val_metrics.get('moving_cells', 0.0)):.1f}"
            )

        if args.benchmark_every > 0 and epoch % args.benchmark_every == 0:
            motion_metrics = benchmark_motion_val(
                model=model,
                loader=val_loader,
                device=device,
                config=config,
            )
            history.setdefault("motion_benchmark", []).append({"epoch": epoch, **motion_metrics})
            print(
                "motion benchmark:"
                f" valid={int(motion_metrics['motion_valid_cells'])}"
                f" L1(mx,my,mw,mh)="
                f"({motion_metrics['motion_l1_mx']:.5f},"
                f" {motion_metrics['motion_l1_my']:.5f},"
                f" {motion_metrics['motion_l1_mw']:.5f},"
                f" {motion_metrics['motion_l1_mh']:.5f})"
                f" centerL2={motion_metrics['motion_center_l2']:.5f}"
                f" futureIoU={motion_metrics['motion_future_iou']:.4f}"
            )
            if "future_head_center_l2" in motion_metrics:
                print(
                    "future-head benchmark:"
                    f" centerL2={motion_metrics['future_head_center_l2']:.5f}"
                    f" futureIoU={motion_metrics['future_head_iou']:.4f}"
                )

        if args.detect_data and args.benchmark_every > 0 and epoch % args.benchmark_every == 0:
            detect_metrics = run_source_detector_benchmark(
                checkpoint=args.checkpoint,
                data_yaml=args.detect_data,
                imgsz=args.detect_imgsz,
                batch=args.detect_batch,
                device=device,
            )
            history.setdefault("source_detector", []).append({"epoch": epoch, **detect_metrics})
            print(
                "source detector benchmark:"
                f" P={detect_metrics['detect_precision']:.4f}"
                f" R={detect_metrics['detect_recall']:.4f}"
                f" mAP50={detect_metrics['detect_map50']:.4f}"
                f" mAP50-95={detect_metrics['detect_map50_95']:.4f}"
            )

        current_val_metric = float(val_metrics.get(best_val_key, val_metrics["loss_motion"]))
        if current_val_metric < best_val_metric - args.min_delta:
            best_val_metric = current_val_metric
            bad_epochs = 0
            best_path = save_best_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                best_val_motion=best_val_metric,
                source_checkpoint=str(args.checkpoint),
            )
            print(f"new best {best_val_key}: {best_val_metric:.4f} | saved: {best_path}")
        else:
            bad_epochs += 1
            print(f"no val motion improvement: bad_epochs={bad_epochs}/{args.patience}")

        save_history(output_dir, history)
        saved_path = save_checkpoint(output_dir, epoch, model, optimizer, history)
        print(f"saved: {saved_path}")

        if bad_epochs >= args.patience:
            print(f"early stopping: best {best_val_key} = {best_val_metric:.4f}")
            break


if __name__ == "__main__":
    main()
