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
from yolo2dt.trainer import run_epoch, save_checkpoint, save_history
from yolo2dt.utils import count_parameters, ensure_dir, set_seed
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
