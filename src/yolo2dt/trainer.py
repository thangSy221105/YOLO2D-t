from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .utils import ensure_dir, move_batch_to_device, unpack_batch


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    train: bool,
    mixed_precision: bool,
    grad_clip_norm: float | None,
    log_interval: int,
) -> Dict[str, float]:
    model.train(mode=train)

    totals: Dict[str, float] | None = None

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)
        images, targets, motion_mask = unpack_batch(batch)

        if train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = mixed_precision and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            preds = model(images)
            loss_dict = criterion(preds, targets, motion_mask)
            loss = loss_dict["loss"]

        if totals is None:
            totals = {key: 0.0 for key in loss_dict.keys()}

        if train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].detach().item())

        if step % max(log_interval, 1) == 0:
            postfix = {"loss": f"{totals['loss'] / step:.4f}"}
            if "loss_future" in totals:
                postfix["future"] = f"{totals['loss_future'] / step:.4f}"
            if "loss_direction" in totals:
                postfix["dir"] = f"{totals['loss_direction'] / step:.4f}"
            if "mean_cosine" in totals:
                postfix["cos"] = f"{totals['mean_cosine'] / step:.4f}"
            iterator.set_postfix(**postfix)

    num_steps = max(len(loader), 1)
    if totals is None:
        return {"loss": 0.0}
    return {key: value / num_steps for key, value in totals.items()}


def save_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    history: Dict[str, list],
) -> Path:
    output_dir = ensure_dir(output_dir)
    checkpoint_path = output_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_history(output_dir: str | Path, history: Dict[str, list]) -> None:
    output_dir = ensure_dir(output_dir)
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
