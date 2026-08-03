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
    parser.add_argument(
        "--resume",
        type=str,
        default="latest",
        help="Checkpoint path or 'latest' to use the newest epoch_*.pt under train.output_dir.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override final total epochs. If omitted, uses train.epochs from config.",
    )
    return parser.parse_args()


def find_latest_checkpoint(output_dir: str | Path) -> Path:
    checkpoints = sorted(Path(output_dir).glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {output_dir}")
    return checkpoints[-1]


def resolve_checkpoint(resume_arg: str, output_dir: str | Path) -> Path:
    if resume_arg == "latest":
        return find_latest_checkpoint(output_dir)
    checkpoint = Path(resume_arg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = ensure_dir(config["train"]["output_dir"])
    checkpoint_path = resolve_checkpoint(args.resume, output_dir)

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
        lambda_coord=config["loss"]["lambda_coord"],
        lambda_noobj=config["loss"]["lambda_noobj"],
        lambda_class=config["loss"]["lambda_class"],
        lambda_motion=config["loss"]["lambda_motion"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epoch = int(checkpoint["epoch"])
    start_epoch = completed_epoch + 1
    total_epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    history = checkpoint.get("history", {"train": [], "val": []})

    mixed_precision = bool(config["train"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    print("model: Yolo2DTiny")
    print(f"device: {device}")
    print(f"resume checkpoint: {checkpoint_path}")
    print(f"completed epoch: {completed_epoch}")
    print(f"target total epochs: {total_epochs}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"trainable params: {count_parameters(model.parameters()):,}")

    if start_epoch > total_epochs:
        print("Nothing to resume: checkpoint epoch is already >= target total epochs.")
        return

    for epoch in range(start_epoch, total_epochs + 1):
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

        save_history(output_dir, history)
        if config["train"].get("save_every_epoch", True):
            saved_path = save_checkpoint(output_dir, epoch, model, optimizer, history)
            print(f"saved: {saved_path}")


if __name__ == "__main__":
    main()
