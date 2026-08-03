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
from yolo2dt.loss import Yolo2DTLoss
from yolo2dt.model import Yolo2DTiny
from yolo2dt.trainer import run_epoch, save_checkpoint, save_history
from yolo2dt.utils import count_parameters, ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Source checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tune epochs to run.")
    parser.add_argument("--output-dir", type=str, default="outputs/motion_finetune")
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--lambda-coord", type=float, default=1.0)
    parser.add_argument("--lambda-noobj", type=float, default=0.1)
    parser.add_argument("--lambda-class", type=float, default=0.2)
    parser.add_argument("--lambda-motion", type=float, default=5.0)
    parser.add_argument("--use-focal-conf", action="store_true")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5, help="Early stop using validation motion loss.")
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
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
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    requested_device = config["train"].get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    train_loader, val_loader = build_dataloaders(config)

    model = Yolo2DTiny(
        in_channels=config["model"]["in_channels"],
        grid_size=config["data"]["grid_size"],
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = Yolo2DTLoss(
        boxes_per_cell=config["data"]["boxes_per_cell"],
        num_classes=config["data"]["num_classes"],
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
        lambda_class=args.lambda_class,
        lambda_motion=args.lambda_motion,
        use_focal_conf=args.use_focal_conf,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config["train"]["weight_decay"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    source_epoch = int(checkpoint.get("epoch", 0))
    history = {"train": [], "val": []}
    best_val_motion = float("inf")
    bad_epochs = 0

    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    print("model: Yolo2DTiny motion fine-tune")
    print(f"device: {device}")
    print(f"source checkpoint: {checkpoint_path}")
    print(f"source epoch: {source_epoch}")
    print(f"fine-tune epochs: {args.epochs}")
    print(f"output dir: {output_dir}")
    print(f"current lr: {current_lr(optimizer):.3e}")
    print(
        "loss weights:"
        f" coord={args.lambda_coord}"
        f" noobj={args.lambda_noobj}"
        f" class={args.lambda_class}"
        f" motion={args.lambda_motion}"
    )
    print(f"focal confidence: {args.use_focal_conf}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"trainable params: {count_parameters(model.parameters()):,}")

    for finetune_epoch in range(1, args.epochs + 1):
        logical_epoch = source_epoch + finetune_epoch
        print(f"\nFine-tune epoch {finetune_epoch}/{args.epochs} | logical epoch {logical_epoch}")

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

        val_motion = float(val_metrics["loss_motion"])
        if val_motion < best_val_motion - args.min_delta:
            best_val_motion = val_motion
            bad_epochs = 0
            best_path = save_best_checkpoint(output_dir, logical_epoch, model, optimizer, history, best_val_motion)
            print(f"new best val motion: {best_val_motion:.4f} | saved: {best_path}")
        else:
            bad_epochs += 1
            print(f"no val motion improvement: bad_epochs={bad_epochs}/{args.patience}")

        save_history(output_dir, history)
        saved_path = save_checkpoint(output_dir, logical_epoch, model, optimizer, history)
        print(f"saved: {saved_path}")

        if bad_epochs >= args.patience:
            print(f"early stopping: best val motion = {best_val_motion:.4f}")
            break


if __name__ == "__main__":
    main()
