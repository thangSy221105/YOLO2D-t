from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo2dt.config import load_config
from yolo2dt.data_adapter import build_dataloaders
from yolo2dt.loss import Yolo2DTLoss
from yolo2dt.trainer import run_epoch, save_checkpoint, save_history
from yolo2dt.utils import box_iou_xywh, count_parameters, ensure_dir, move_batch_to_device, set_seed, unpack_batch
from yolo2dt.yolov8_motion import YoloV8MotionAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolov8_motion.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="YOLOv8 detect-only checkpoint (.pt).")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override train.output_dir from config.")
    parser.add_argument("--lr", type=float, default=None, help="Override train.lr from config.")
    parser.add_argument("--lambda-coord", type=float, default=None)
    parser.add_argument("--lambda-noobj", type=float, default=None)
    parser.add_argument("--lambda-class", type=float, default=None)
    parser.add_argument("--lambda-motion", type=float, default=None)
    parser.add_argument("--unfreeze-detector", action="store_true", help="Train the full YOLOv8 detector too.")
    parser.add_argument(
        "--freeze-first-conv",
        action="store_true",
        help="Keep the upgraded 6-channel first conv frozen. Default behavior is trainable.",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stop based on total validation loss.")
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--benchmark-every", type=int, default=1, help="Run val benchmark every N epochs. Use 0 to disable.")
    parser.add_argument("--benchmark-conf", type=float, default=0.25)
    parser.add_argument("--benchmark-iou", type=float, default=0.5)
    return parser.parse_args()


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_best_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict[str, list],
    best_val_loss: float,
    source_checkpoint: str,
) -> Path:
    output_dir = ensure_dir(output_dir)
    checkpoint_path = output_dir / "best.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
            "source_checkpoint": source_checkpoint,
        },
        checkpoint_path,
    )
    return checkpoint_path


def xywh_dict_to_tensor(box: dict[str, float]) -> torch.Tensor:
    return torch.tensor([box["xc"], box["yc"], box["w"], box["h"]], dtype=torch.float32)


def decode_gt(
    target: torch.Tensor,
    motion_mask: torch.Tensor,
    grid_size: int,
    boxes_per_cell: int,
    num_classes: int,
    conf_thr: float = 0.5,
) -> list[dict[str, float]]:
    motion_offset = boxes_per_cell * 5 + num_classes
    target = target.detach().cpu()
    motion_mask = motion_mask.detach().cpu()
    boxes: list[dict[str, float]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            conf = float(target[row, col, 4])
            if conf < conf_thr:
                continue

            x_cell = float(target[row, col, 0])
            y_cell = float(target[row, col, 1])
            w = float(target[row, col, 2])
            h = float(target[row, col, 3])
            motion = target[row, col, motion_offset : motion_offset + 4]

            boxes.append(
                {
                    "xc": (col + x_cell) / grid_size,
                    "yc": (row + y_cell) / grid_size,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "motion_valid": float(motion_mask[row, col]),
                    "mx": float(motion[0]),
                    "my": float(motion[1]),
                    "mw": float(motion[2]),
                    "mh": float(motion[3]),
                }
            )
    return boxes


def decode_pred(
    pred: torch.Tensor,
    grid_size: int,
    boxes_per_cell: int,
    num_classes: int,
    conf_thr: float,
) -> list[dict[str, float]]:
    motion_offset = boxes_per_cell * 5 + num_classes
    pred = pred.detach().cpu()
    boxes: list[dict[str, float]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            motion = pred[row, col, motion_offset : motion_offset + 4]
            for box_idx in range(boxes_per_cell):
                offset = box_idx * 5
                conf = float(pred[row, col, offset + 4])
                if conf < conf_thr:
                    continue

                x_cell = float(pred[row, col, offset])
                y_cell = float(pred[row, col, offset + 1])
                boxes.append(
                    {
                        "xc": (col + x_cell) / grid_size,
                        "yc": (row + y_cell) / grid_size,
                        "w": max(float(pred[row, col, offset + 2]), 1.0e-6),
                        "h": max(float(pred[row, col, offset + 3]), 1.0e-6),
                        "conf": conf,
                        "mx": float(motion[0]),
                        "my": float(motion[1]),
                        "mw": float(motion[2]),
                        "mh": float(motion[3]),
                    }
                )
    return boxes


def nms_boxes(boxes: list[dict[str, float]], iou_thr: float) -> list[dict[str, float]]:
    boxes = sorted(boxes, key=lambda box: box["conf"], reverse=True)
    keep: list[dict[str, float]] = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        best_t = xywh_dict_to_tensor(best).unsqueeze(0)
        remaining = []
        for box in boxes:
            box_t = xywh_dict_to_tensor(box).unsqueeze(0)
            iou = float(box_iou_xywh(best_t, box_t)[0])
            if iou < iou_thr:
                remaining.append(box)
        boxes = remaining
    return keep


def compute_ap_11pt(recalls: np.ndarray, precisions: np.ndarray) -> float:
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        valid = precisions[recalls >= threshold]
        ap += (float(valid.max()) if len(valid) else 0.0) / 11.0
    return ap


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xc, yc, w, h = box
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
def benchmark_val(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict,
    conf_thr: float,
    iou_thr: float,
) -> dict[str, float]:
    model.eval()
    grid_size = int(config["data"]["grid_size"])
    boxes_per_cell = int(config["data"]["boxes_per_cell"])
    num_classes = int(config["data"]["num_classes"])

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gts = 0
    all_ious: list[float] = []
    all_predictions: list[dict[str, object]] = []
    image_index = 0

    motion_count = 0
    motion_l1 = torch.zeros(4, dtype=torch.float64)
    motion_center_l2 = 0.0
    motion_future_iou = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images, targets, motion_masks = unpack_batch(batch)
        preds = model(images)

        for sample_idx in range(images.shape[0]):
            gt_boxes = decode_gt(
                targets[sample_idx],
                motion_masks[sample_idx],
                grid_size=grid_size,
                boxes_per_cell=boxes_per_cell,
                num_classes=num_classes,
            )
            pred_boxes = decode_pred(
                preds[sample_idx],
                grid_size=grid_size,
                boxes_per_cell=boxes_per_cell,
                num_classes=num_classes,
                conf_thr=conf_thr,
            )
            pred_boxes = nms_boxes(pred_boxes, iou_thr=iou_thr)
            total_gts += len(gt_boxes)

            matched_gt: set[int] = set()
            for pred in sorted(pred_boxes, key=lambda box: box["conf"], reverse=True):
                pred_t = xywh_dict_to_tensor(pred).unsqueeze(0)
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    gt_t = xywh_dict_to_tensor(gt).unsqueeze(0)
                    iou = float(box_iou_xywh(pred_t, gt_t)[0])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                all_predictions.append(
                    {
                        "image_id": image_index,
                        "conf": float(pred["conf"]),
                        "pred": pred,
                        "gt_boxes": gt_boxes,
                    }
                )

                if best_iou >= iou_thr:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    all_ious.append(best_iou)

                    gt = gt_boxes[best_gt_idx]
                    if gt["motion_valid"] > 0.5:
                        motion_count += 1
                        motion_l1 += torch.tensor(
                            [
                                abs(pred["mx"] - gt["mx"]),
                                abs(pred["my"] - gt["my"]),
                                abs(pred["mw"] - gt["mw"]),
                                abs(pred["mh"] - gt["mh"]),
                            ],
                            dtype=torch.float64,
                        )
                        motion_center_l2 += (
                            (pred["mx"] - gt["mx"]) ** 2 + (pred["my"] - gt["my"]) ** 2
                        ) ** 0.5

                        gt_future = xywh_to_xyxy(
                            (
                                gt["xc"] + gt["mx"],
                                gt["yc"] + gt["my"],
                                max(gt["w"] + gt["mw"], 1.0e-6),
                                max(gt["h"] + gt["mh"], 1.0e-6),
                            )
                        )
                        pred_future = xywh_to_xyxy(
                            (
                                pred["xc"] + pred["mx"],
                                pred["yc"] + pred["my"],
                                max(pred["w"] + pred["mw"], 1.0e-6),
                                max(pred["h"] + pred["mh"], 1.0e-6),
                            )
                        )
                        motion_future_iou += box_iou_single(gt_future, pred_future)
                else:
                    total_fp += 1

            total_fn += len(gt_boxes) - len(matched_gt)
            image_index += 1

    all_predictions = sorted(all_predictions, key=lambda item: float(item["conf"]), reverse=True)
    tp_curve = np.zeros(len(all_predictions), dtype=np.float64)
    fp_curve = np.zeros(len(all_predictions), dtype=np.float64)
    matched_for_ap: dict[int, set[int]] = {}

    for pred_idx, pred_item in enumerate(all_predictions):
        image_id = int(pred_item["image_id"])
        pred = pred_item["pred"]
        gt_boxes = pred_item["gt_boxes"]
        pred_t = xywh_dict_to_tensor(pred).unsqueeze(0)
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            gt_t = xywh_dict_to_tensor(gt).unsqueeze(0)
            iou = float(box_iou_xywh(pred_t, gt_t)[0])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr:
            matched_for_ap.setdefault(image_id, set())
            if best_gt_idx not in matched_for_ap[image_id]:
                tp_curve[pred_idx] = 1.0
                matched_for_ap[image_id].add(best_gt_idx)
            else:
                fp_curve[pred_idx] = 1.0
        else:
            fp_curve[pred_idx] = 1.0

    cum_tp = np.cumsum(tp_curve)
    cum_fp = np.cumsum(fp_curve)
    recalls = cum_tp / max(total_gts, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1.0e-12)
    motion_mean_l1 = motion_l1 / max(motion_count, 1)

    return {
        "bench_precision": total_tp / max(total_tp + total_fp, 1),
        "bench_recall": total_tp / max(total_tp + total_fn, 1),
        "bench_f1": (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1),
        "bench_mean_iou": sum(all_ious) / max(len(all_ious), 1),
        "bench_ap50": compute_ap_11pt(recalls, precisions),
        "bench_tp": float(total_tp),
        "bench_fp": float(total_fp),
        "bench_fn": float(total_fn),
        "motion_valid_matches": float(motion_count),
        "motion_l1_mx": float(motion_mean_l1[0]),
        "motion_l1_my": float(motion_mean_l1[1]),
        "motion_l1_mw": float(motion_mean_l1[2]),
        "motion_l1_mh": float(motion_mean_l1[3]),
        "motion_center_l2": motion_center_l2 / max(motion_count, 1),
        "motion_future_iou": motion_future_iou / max(motion_count, 1),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = ensure_dir(args.output_dir or config["train"]["output_dir"])

    requested_device = config["train"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    train_loader, val_loader = build_dataloaders(config)

    model = YoloV8MotionAdapter(
        checkpoint_path=checkpoint_path,
        image_size=config["data"]["image_size"],
        grid_size=config["data"]["grid_size"],
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        hidden_dim=config["model"]["hidden_dim"],
        freeze_detector=not args.unfreeze_detector,
        train_first_conv=not args.freeze_first_conv,
    ).to(device)

    criterion = Yolo2DTLoss(
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        lambda_coord=float(args.lambda_coord if args.lambda_coord is not None else config["loss"]["lambda_coord"]),
        lambda_noobj=float(args.lambda_noobj if args.lambda_noobj is not None else config["loss"]["lambda_noobj"]),
        lambda_class=float(args.lambda_class if args.lambda_class is not None else config["loss"]["lambda_class"]),
        lambda_motion=float(args.lambda_motion if args.lambda_motion is not None else config["loss"]["lambda_motion"]),
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr if args.lr is not None else config["train"]["lr"]),
        weight_decay=config["train"]["weight_decay"],
    )

    total_epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    bad_epochs = 0

    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    print("model: YoloV8MotionAdapter (joint detect + motion)")
    print(f"device: {device}")
    print(f"source checkpoint: {checkpoint_path}")
    print(f"fine-tune epochs: {total_epochs}")
    print(f"output dir: {output_dir}")
    print(f"current lr: {current_lr(optimizer):.3e}")
    print(f"freeze detector: {not args.unfreeze_detector}")
    print(f"train first conv: {not args.freeze_first_conv}")
    print(
        "loss weights:"
        f" coord={criterion.lambda_coord}"
        f" noobj={criterion.lambda_noobj}"
        f" class={criterion.lambda_class}"
        f" motion={criterion.lambda_motion}"
    )
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"trainable params: {count_parameters(model.parameters()):,}")

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
        print(
            "train detail:"
            f" coord={train_metrics['loss_coord']:.4f}"
            f" obj={train_metrics['loss_obj']:.4f}"
            f" noobj={train_metrics['loss_noobj']:.4f}"
            f" cls={train_metrics['loss_cls']:.4f}"
            f" motion={train_metrics['loss_motion']:.4f}"
        )
        print(
            "val detail:"
            f" coord={val_metrics['loss_coord']:.4f}"
            f" obj={val_metrics['loss_obj']:.4f}"
            f" noobj={val_metrics['loss_noobj']:.4f}"
            f" cls={val_metrics['loss_cls']:.4f}"
            f" motion={val_metrics['loss_motion']:.4f}"
        )

        benchmark_metrics = None
        if args.benchmark_every > 0 and epoch % args.benchmark_every == 0:
            benchmark_metrics = benchmark_val(
                model=model,
                loader=val_loader,
                device=device,
                config=config,
                conf_thr=args.benchmark_conf,
                iou_thr=args.benchmark_iou,
            )
            history.setdefault("benchmark", []).append({"epoch": epoch, **benchmark_metrics})
            print(
                "val benchmark:"
                f" P={benchmark_metrics['bench_precision']:.4f}"
                f" R={benchmark_metrics['bench_recall']:.4f}"
                f" F1={benchmark_metrics['bench_f1']:.4f}"
                f" AP50={benchmark_metrics['bench_ap50']:.4f}"
                f" meanIoU={benchmark_metrics['bench_mean_iou']:.4f}"
            )
            print(
                "motion benchmark:"
                f" valid={int(benchmark_metrics['motion_valid_matches'])}"
                f" L1(mx,my,mw,mh)="
                f"({benchmark_metrics['motion_l1_mx']:.5f},"
                f" {benchmark_metrics['motion_l1_my']:.5f},"
                f" {benchmark_metrics['motion_l1_mw']:.5f},"
                f" {benchmark_metrics['motion_l1_mh']:.5f})"
                f" centerL2={benchmark_metrics['motion_center_l2']:.5f}"
                f" futureIoU={benchmark_metrics['motion_future_iou']:.4f}"
            )

        val_loss = float(val_metrics["loss"])
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            bad_epochs = 0
            best_path = save_best_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                best_val_loss=best_val_loss,
                source_checkpoint=str(checkpoint_path),
            )
            print(f"new best val loss: {best_val_loss:.4f} | saved: {best_path}")
        else:
            bad_epochs += 1
            print(f"no val improvement: bad_epochs={bad_epochs}/{args.patience}")

        save_history(output_dir, history)
        saved_path = save_checkpoint(output_dir, epoch, model, optimizer, history)
        print(f"saved: {saved_path}")

        if bad_epochs >= args.patience:
            print(f"early stopping: best val loss = {best_val_loss:.4f}")
            break


if __name__ == "__main__":
    main()
