from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


VOC_CLASSES = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]

VOC_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(VOC_CLASSES)}


class VocCsvDetectionDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: int = 448,
        grid_size: int = 7,
        num_classes: int = 20,
        normalize_mean: List[float] | None = None,
        normalize_std: List[float] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.return_meta = return_meta
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        split_dir = self.root_dir / split
        self.image_dir = split_dir / "images"
        self.target_dir = split_dir / "targets"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {self.image_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Missing target dir: {self.target_dir}")

        self.ids = sorted(path.stem for path in self.image_dir.glob("*.jpg"))
        if not self.ids:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_boxes(self, target_path: Path) -> List[Dict[str, float | int]]:
        boxes: List[Dict[str, float | int]] = []
        with open(target_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["object"]
                if label not in VOC_CLASS_TO_INDEX:
                    continue
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
                boxes.append(
                    {
                        "cls": VOC_CLASS_TO_INDEX[label],
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )
        return boxes

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        image_path = self.image_dir / f"{sample_id}.jpg"
        target_path = self.target_dir / f"{sample_id}.csv"

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image = TF.resize(image, [self.image_size, self.image_size])
        image = TF.to_tensor(image)

        if self.normalize_mean is not None and self.normalize_std is not None:
            image = TF.normalize(image, self.normalize_mean, self.normalize_std)

        target = torch.zeros(self.grid_size, self.grid_size, 1 + self.num_classes + 4, dtype=torch.float32)
        boxes = self._load_boxes(target_path)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        for box in boxes:
            xmin = float(box["xmin"]) * scale_x
            ymin = float(box["ymin"]) * scale_y
            xmax = float(box["xmax"]) * scale_x
            ymax = float(box["ymax"]) * scale_y
            cls_id = int(box["cls"])

            xc = ((xmin + xmax) / 2.0) / self.image_size
            yc = ((ymin + ymax) / 2.0) / self.image_size
            w = (xmax - xmin) / self.image_size
            h = (ymax - ymin) / self.image_size

            col = min(self.grid_size - 1, max(0, int(xc * self.grid_size)))
            row = min(self.grid_size - 1, max(0, int(yc * self.grid_size)))

            cell_x = xc * self.grid_size - col
            cell_y = yc * self.grid_size - row

            # If multiple objects land in the same cell, keep the larger one.
            if target[row, col, 0] > 0:
                prev_area = float(target[row, col, 1 + self.num_classes + 2] * target[row, col, 1 + self.num_classes + 3])
                if prev_area >= w * h:
                    continue

            target[row, col].zero_()
            target[row, col, 0] = 1.0
            target[row, col, 1 + cls_id] = 1.0
            target[row, col, 1 + self.num_classes : 1 + self.num_classes + 4] = torch.tensor(
                [cell_x, cell_y, w, h], dtype=torch.float32
            )

        if not self.return_meta:
            return image, target

        meta = {"id": sample_id, "image_path": str(image_path)}
        return {"image": image, "target": target, "meta": meta}
