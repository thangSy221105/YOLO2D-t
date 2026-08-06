from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo2dt.config import load_config
from yolo2dt.utils import count_parameters, ensure_dir, set_seed
from yolo2dt.voc_yolov1_dataset import VOC_CLASSES, VocCsvDetectionDataset
from yolo2dt.yolov1_loss import YoloV1Loss
from yolo2dt.yolov1_model import YoloV1Original


def cxcywh_to_xyxy(box):
    xc, yc, w, h = box
    return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dtype=np.float32)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        remain = []
        for j in order[1:]:
            if iou_xyxy(boxes[i], boxes[j]) < iou_thr:
                remain.append(j)
        order = np.array(remain, dtype=np.int64)
    return keep


def decode_person_predictions(pred, grid_size, num_classes, boxes_per_cell, conf_thr, nms_iou):
    pred = pred.detach().cpu().numpy()
    outputs = []

    for sample in pred:
        sample_preds = []
        cls_scores = sample[..., :num_classes]

        for r in range(grid_size):
            for c in range(grid_size):
                cls_id = int(np.argmax(cls_scores[r, c]))
                if cls_id != 0:
                    continue

                cls_prob = float(cls_scores[r, c, cls_id])
                for b in range(boxes_per_cell):
                    off = num_classes + b * 5
                    conf = float(sample[r, c, off + 0])
                    x = float(sample[r, c, off + 1])
                    y = float(sample[r, c, off + 2])
                    sqrtw = float(sample[r, c, off + 3])
                    sqrth = float(sample[r, c, off + 4])

                    score = conf * cls_prob
                    if score < conf_thr:
                        continue

                    xc = (c + x) / grid_size
                    yc = (r + y) / grid_size
                    w = max(sqrtw * sqrtw, 1.0e-6)
                    h = max(sqrth * sqrth, 1.0e-6)
                    sample_preds.append({"score": score, "xyxy": cxcywh_to_xyxy([xc, yc, w, h]).clip(0.0, 1.0)})

        if not sample_preds:
            outputs.append([])
            continue

        boxes = np.stack([item["xyxy"] for item in sample_preds])
        scores = np.array([item["score"] for item in sample_preds], dtype=np.float32)
        keep = nms(boxes, scores, iou_thr=nms_iou)
        outputs.append([sample_preds[idx] for idx in keep])

    return outputs


def decode_person_targets(target, grid_size, num_classes):
    target = target.detach().cpu().numpy()
    outputs = []

    for sample in target:
        sample_targets = []
        for r in range(grid_size):
            for c in range(grid_size):
                obj = float(sample[r, c, 0])
                if obj <= 0:
                    continue

                cls_id = int(np.argmax(sample[r, c, 1 : num_classes + 1]))
                if cls_id != 0:
                    continue

                x = float(sample[r, c, num_classes + 1])
                y = float(sample[r, c, num_classes + 2])
                w = float(sample[r, c, num_classes + 3])
                h = float(sample[r, c, num_classes + 4])

                xc = (c + x) / grid_size
                yc = (r + y) / grid_size
                sample_targets.append({"xyxy": cxcywh_to_xyxy([xc, yc, w, h]).clip(0.0, 1.0), "matched": False})
        outputs.append(sample_targets)

    return outputs


def compute_ap(recalls, precisions):
    recalls = np.array([0.0] + recalls + [1.0])
    precisions = np.array([0.0] + precisions + [0.0])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


@torch.no_grad()
def benchmark_person(model, loader, device, grid_size, num_classes, boxes_per_cell, conf_thr, nms_iou, eval_iou):
    model.eval()

    total_gt = 0
    all_preds = []
    all_gts = {}
    image_id = 0

    for images, targets in tqdm(loader, desc="person benchmark", leave=False):
        images = images.to(device, non_blocking=True)
        preds = model(images)

        batch_preds = decode_person_predictions(preds, grid_size, num_classes, boxes_per_cell, conf_thr, nms_iou)
        batch_gts = decode_person_targets(targets, grid_size, num_classes)

        for pred_list, gt_list in zip(batch_preds, batch_gts):
            all_gts[image_id] = gt_list
            total_gt += len(gt_list)
            for pred in pred_list:
                all_preds.append({"image_id": image_id, "score": pred["score"], "xyxy": pred["xyxy"]})
            image_id += 1

    all_preds = sorted(all_preds, key=lambda x: x["score"], reverse=True)

    tp, fp = [], []
    for pred in all_preds:
        gts = all_gts[pred["image_id"]]
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if gt["matched"]:
                continue
            current_iou = iou_xyxy(pred["xyxy"], gt["xyxy"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        if best_iou >= eval_iou and best_idx >= 0:
            gts[best_idx]["matched"] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = tp_cum / max(total_gt, 1)
    ap50 = compute_ap(recalls.tolist(), precisions.tolist()) if len(tp) else 0.0

    final_tp = int(tp_cum[-1]) if len(tp_cum) else 0
    final_fp = int(fp_cum[-1]) if len(fp_cum) else 0
    final_precision = final_tp / max(final_tp + final_fp, 1)
    final_recall = final_tp / max(total_gt, 1)

    return {
        "gt": total_gt,
        "pred": len(all_preds),
        "tp": final_tp,
        "fp": final_fp,
        "precision": final_precision,
        "recall": final_recall,
        "ap50": ap50,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolov1_voc.yaml")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, mixed_precision, log_interval):
    model.train(mode=train)

    totals = {
        "loss": 0.0,
        "loss_coord": 0.0,
        "loss_obj": 0.0,
        "loss_noobj": 0.0,
        "loss_cls": 0.0,
        "loss_motion": 0.0,
    }

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for step, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = mixed_precision and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            preds = model(images)
            loss_dict = criterion(preds, targets)
            loss = loss_dict["loss"]

        if train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].detach().item())

        if step % max(log_interval, 1) == 0:
            iterator.set_postfix(loss=f"{totals['loss'] / step:.4f}")

    num_steps = max(len(loader), 1)
    return {key: value / num_steps for key, value in totals.items()}


def build_loaders(config: dict):
    data_cfg = config["data"]
    normalize_mean = data_cfg.get("normalize_mean")
    normalize_std = data_cfg.get("normalize_std")

    train_dataset = VocCsvDetectionDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg["train_split"],
        image_size=data_cfg["image_size"],
        grid_size=data_cfg["grid_size"],
        num_classes=data_cfg["num_classes"],
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    val_dataset = VocCsvDetectionDataset(
        root_dir=data_cfg["root_dir"],
        split=data_cfg["val_split"],
        image_size=data_cfg["image_size"],
        grid_size=data_cfg["grid_size"],
        num_classes=data_cfg["num_classes"],
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader


def save_checkpoint(output_dir: Path, epoch: int, model, optimizer, history, best_val: float):
    checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = ensure_dir(config["train"]["output_dir"])
    requested_device = config["train"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    train_loader, val_loader = build_loaders(config)

    model = YoloV1Original(
        grid_size=config["data"]["grid_size"],
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
    ).to(device)

    criterion = YoloV1Loss(
        grid_size=config["data"]["grid_size"],
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        lambda_coord=config["loss"]["lambda_coord"],
        lambda_noobj=config["loss"]["lambda_noobj"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    start_epoch = 1
    history = {"train": [], "val": []}
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", history)
        best_val = checkpoint.get("best_val_loss", best_val)
        start_epoch = int(checkpoint["epoch"]) + 1

    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    print(f"device: {device}")
    print(f"classes: {len(VOC_CLASSES)}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"trainable params: {count_parameters(model.parameters()):,}")

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            mixed_precision=mixed_precision,
            log_interval=config["train"].get("log_interval", 50),
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
            log_interval=config["train"].get("log_interval", 50),
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        person_metrics = benchmark_person(
            model=model,
            loader=val_loader,
            device=device,
            grid_size=config["data"]["grid_size"],
            num_classes=config["data"]["num_classes"],
            boxes_per_cell=config["data"]["boxes_per_cell"],
            conf_thr=config["eval"].get("conf_threshold", 0.05),
            nms_iou=config["eval"].get("nms_iou", 0.5),
            eval_iou=config["eval"].get("eval_iou", 0.5),
        )
        history.setdefault("person", []).append(person_metrics)

        print(f"train loss: {train_metrics['loss']:.4f} | val loss: {val_metrics['loss']:.4f}")
        print(
            "train detail:"
            f" coord={train_metrics['loss_coord']:.4f}"
            f" obj={train_metrics['loss_obj']:.4f}"
            f" noobj={train_metrics['loss_noobj']:.4f}"
            f" cls={train_metrics['loss_cls']:.4f}"
        )
        print(
            "person benchmark:"
            f" gt={person_metrics['gt']}"
            f" pred={person_metrics['pred']}"
            f" P={person_metrics['precision']:.4f}"
            f" R={person_metrics['recall']:.4f}"
            f" AP50={person_metrics['ap50']:.4f}"
        )

        save_path = save_checkpoint(output_dir, epoch, model, optimizer, history, best_val)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = output_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_loss": best_val,
                },
                best_path,
            )
            print(f"new best val loss: {best_val:.4f} | saved: {best_path}")

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
