from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

        print(f"train loss: {train_metrics['loss']:.4f} | val loss: {val_metrics['loss']:.4f}")
        print(
            "train detail:"
            f" coord={train_metrics['loss_coord']:.4f}"
            f" obj={train_metrics['loss_obj']:.4f}"
            f" noobj={train_metrics['loss_noobj']:.4f}"
            f" cls={train_metrics['loss_cls']:.4f}"
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
