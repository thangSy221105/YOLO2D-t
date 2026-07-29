from __future__ import annotations

import importlib
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class Dummy2DTDataset(Dataset):
    def __init__(self, length: int = 32, image_size: int = 448, grid_size: int = 7, output_dim: int = 15):
        self.length = length
        self.image_size = image_size
        self.grid_size = grid_size
        self.output_dim = output_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image = torch.rand(6, self.image_size, self.image_size)
        target = torch.zeros(self.grid_size, self.grid_size, self.output_dim)
        motion_mask = torch.zeros(self.grid_size, self.grid_size)

        row = index % self.grid_size
        col = (index // self.grid_size) % self.grid_size
        target[row, col, 0:5] = torch.tensor([0.5, 0.5, 0.2, 0.25, 1.0])
        target[row, col, 5:10] = torch.tensor([0.5, 0.5, 0.2, 0.25, 1.0])
        target[row, col, 10] = 1.0
        target[row, col, 11:15] = torch.tensor([0.02, 0.01, 0.0, 0.0])
        motion_mask[row, col] = 1.0

        return {"image": image, "target": target, "motion_mask": motion_mask}


def _try_build_external_dataloader(config: dict, split: str):
    try:
        dataset_module = importlib.import_module("scripts.mot17_2dt_dataset")
    except ModuleNotFoundError:
        return None

    if not hasattr(dataset_module, "create_dataloader"):
        return None

    data_cfg = config["data"]
    return dataset_module.create_dataloader(
        processed_dir=data_cfg["processed_dir"],
        split=split,
        delta=data_cfg["delta"],
        image_size=data_cfg["image_size"],
        grid_size=data_cfg["grid_size"],
        boxes_per_cell=data_cfg["boxes_per_cell"],
        batch_size=data_cfg["batch_size"],
        shuffle=(split == data_cfg.get("train_split", "train")),
        num_workers=data_cfg["num_workers"],
        return_meta=data_cfg.get("return_meta", False),
        horizontal_flip=data_cfg.get("use_horizontal_flip", False) if split == data_cfg.get("train_split", "train") else False,
    )


def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    train_loader = _try_build_external_dataloader(config, config["data"]["train_split"])
    val_loader = _try_build_external_dataloader(config, config["data"]["val_split"])

    if train_loader is not None and val_loader is not None:
        return train_loader, val_loader

    output_dim = config["data"]["boxes_per_cell"] * 5 + config["data"]["num_classes"] + 4
    train_dataset = Dummy2DTDataset(
        length=64,
        image_size=config["data"]["image_size"],
        grid_size=config["data"]["grid_size"],
        output_dim=output_dim,
    )
    val_dataset = Dummy2DTDataset(
        length=16,
        image_size=config["data"]["image_size"],
        grid_size=config["data"]["grid_size"],
        output_dim=output_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"].get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"].get("pin_memory", True),
    )
    return train_loader, val_loader
