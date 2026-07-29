"""PyTorch Dataset for preprocessed MOT17 YOLOv1 2D+t samples."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset


CONFIG = {
    "processed_dir": Path("data/processed"),
    "split": "train",
    "delta": 40,
    "image_size": 448,
    "grid_size": 7,
    "boxes_per_cell": 2,
    "batch_size": 4,
    "num_workers": 0,
    "shuffle": True,
    "horizontal_flip": False,
    "visualize": True,
    "visualize_path": Path("data/processed/visualizations/batch_train_delta40.jpg"),
}


class MOT17TrajectoryDataset(Dataset):
    """Load two MOT17 frames and build a YOLOv1-style trajectory target.

    Target layout is ``S x S x (B * 5 + C + 4)``. With defaults this is
    ``7 x 7 x 15``:

    - box slots: ``x_cell, y_cell, w, h, conf`` repeated ``B`` times
    - class slot: person class probability
    - motion slots: ``mx, my, mw, mh``

    Width, height, and motion are normalized by ``image_size`` after letterbox.
    Motion is shared per grid cell and should be supervised with the returned
    ``motion_mask`` because objects may disappear before ``t + delta``.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        split: str = "train",
        delta: int = 4,
        image_size: int = 448,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 1,
        return_meta: bool = False,
        horizontal_flip: bool = False,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.project_root = self.processed_dir.resolve().parent.parent
        self.split = split
        self.delta = delta
        self.image_size = image_size
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.return_meta = return_meta
        self.horizontal_flip = horizontal_flip

        pair_path = self.processed_dir / "pairs" / f"{split}_delta{delta}.csv"
        if not pair_path.exists():
            raise FileNotFoundError(
                f"Missing pair index {pair_path}. Run scripts/preprocess_mot17_2dt.py first."
            )

        self.pairs = self._read_csv(pair_path)
        self.annotations = self._load_annotations()
        self.target_depth = boxes_per_cell * 5 + num_classes + 4

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, Any]:
        pair = self.pairs[index]
        sequence = pair["sequence"]
        frame_t = int(pair["frame_t"])
        frame_next = int(pair["frame_next"])

        image_t = Image.open(self._resolve_image_path(pair["img_t"])).convert("RGB")
        image_next = Image.open(self._resolve_image_path(pair["img_next"])).convert("RGB")
        orig_w, orig_h = image_t.size

        image_t, scale, pad_x, pad_y = self._letterbox(image_t)
        image_next, _, _, _ = self._letterbox(image_next)

        anns_t = self.annotations[sequence].get(frame_t, {})
        anns_next = self.annotations[sequence].get(frame_next, {})
        objects = self._build_objects(anns_t, anns_next, scale, pad_x, pad_y)

        if self.horizontal_flip and torch.rand(()) < 0.5:
            image_t = ImageOps.mirror(image_t)
            image_next = ImageOps.mirror(image_next)
            objects = [self._flip_object(obj) for obj in objects]

        image = torch.cat([self._to_tensor(image_t), self._to_tensor(image_next)], dim=0)
        target, motion_mask, assigned = self._encode_target(objects)

        if self.return_meta:
            return {
                "image": image,
                "target": target,
                "motion_mask": motion_mask,
                "meta": {
                    "sequence": sequence,
                    "frame_t": frame_t,
                    "frame_next": frame_next,
                    "orig_size": (orig_w, orig_h),
                    "assigned_objects": assigned,
                    "total_objects_t": len(objects),
                },
            }
        return image, target, motion_mask

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _resolve_image_path(self, value: str) -> Path:
        raw_value = str(value).strip()
        path = Path(raw_value)
        if path.is_absolute() or path.exists():
            return path

        # CSV files may be generated on Windows and then consumed on Colab/Linux.
        # Rebuild the path from Windows-style parts so `data\\...` resolves correctly.
        normalized = Path(*PureWindowsPath(raw_value).parts)
        candidate = self.project_root / normalized
        if candidate.exists():
            return candidate

        # Fallback for CSVs that stored paths relative to the processed directory.
        candidate = self.processed_dir.parent / normalized
        if candidate.exists():
            return candidate

        return self.project_root / normalized

    def _load_annotations(self) -> dict[str, dict[int, dict[int, dict[str, float]]]]:
        split_sequences = self.processed_dir / "splits" / f"{self.split}_sequences.txt"
        sequences = [
            line.strip()
            for line in split_sequences.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        annotations: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
        for sequence in sequences:
            path = self.processed_dir / "annotations" / f"{sequence}.csv"
            rows = self._read_csv(path)
            by_frame: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
            for row in rows:
                frame = int(row["frame"])
                track_id = int(row["track_id"])
                by_frame[frame][track_id] = {
                    "track_id": float(track_id),
                    "xc": float(row["xc"]),
                    "yc": float(row["yc"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "visibility": float(row["visibility"]),
                    "class_id": float(row["class_id"]),
                }
            annotations[sequence] = dict(by_frame)
        return annotations

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, float, float, float]:
        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        pad_x = (self.image_size - new_w) / 2.0
        pad_y = (self.image_size - new_h) / 2.0

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def _build_objects(
        self,
        anns_t: dict[int, dict[str, float]],
        anns_next: dict[int, dict[str, float]],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[dict[str, float]]:
        objects: list[dict[str, float]] = []
        for track_id, ann_t in anns_t.items():
            x = ann_t["xc"] * scale + pad_x
            y = ann_t["yc"] * scale + pad_y
            w = ann_t["w"] * scale
            h = ann_t["h"] * scale

            motion_valid = 0.0
            mx = my = mw = mh = 0.0
            if track_id in anns_next:
                ann_next = anns_next[track_id]
                next_x = ann_next["xc"] * scale + pad_x
                next_y = ann_next["yc"] * scale + pad_y
                next_w = ann_next["w"] * scale
                next_h = ann_next["h"] * scale
                mx = next_x - x
                my = next_y - y
                mw = next_w - w
                mh = next_h - h
                motion_valid = 1.0

            objects.append(
                {
                    "track_id": float(track_id),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "mx": mx,
                    "my": my,
                    "mw": mw,
                    "mh": mh,
                    "visibility": ann_t["visibility"],
                    "motion_valid": motion_valid,
                }
            )
        return objects

    def _flip_object(self, obj: dict[str, float]) -> dict[str, float]:
        flipped = dict(obj)
        flipped["x"] = self.image_size - obj["x"]
        flipped["mx"] = -obj["mx"]
        return flipped

    def _encode_target(
        self,
        objects: list[dict[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        target = torch.zeros((self.grid_size, self.grid_size, self.target_depth), dtype=torch.float32)
        motion_mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)

        # YOLOv1 can represent one object per cell. Prefer more visible/larger boxes.
        candidates: dict[tuple[int, int], dict[str, float]] = {}
        for obj in objects:
            x_norm = obj["x"] / self.image_size
            y_norm = obj["y"] / self.image_size
            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                continue

            col = min(self.grid_size - 1, max(0, int(x_norm * self.grid_size)))
            row = min(self.grid_size - 1, max(0, int(y_norm * self.grid_size)))
            key = (row, col)
            prev = candidates.get(key)
            score = obj["visibility"] * max(obj["w"] * obj["h"], 1.0)
            prev_score = -1.0 if prev is None else prev["visibility"] * max(prev["w"] * prev["h"], 1.0)
            if score > prev_score:
                candidates[key] = obj

        box_stride = 5
        class_offset = self.boxes_per_cell * box_stride
        motion_offset = class_offset + self.num_classes

        for (row, col), obj in candidates.items():
            x_norm = obj["x"] / self.image_size
            y_norm = obj["y"] / self.image_size
            x_cell = x_norm * self.grid_size - col
            y_cell = y_norm * self.grid_size - row
            w_norm = obj["w"] / self.image_size
            h_norm = obj["h"] / self.image_size

            for box_idx in range(self.boxes_per_cell):
                offset = box_idx * box_stride
                target[row, col, offset : offset + 5] = torch.tensor(
                    [x_cell, y_cell, w_norm, h_norm, 1.0],
                    dtype=torch.float32,
                )

            target[row, col, class_offset] = 1.0
            target[row, col, motion_offset : motion_offset + 4] = torch.tensor(
                [
                    obj["mx"] / self.image_size,
                    obj["my"] / self.image_size,
                    obj["mw"] / self.image_size,
                    obj["mh"] / self.image_size,
                ],
                dtype=torch.float32,
            )
            motion_mask[row, col] = obj["motion_valid"]

        return target, motion_mask, len(candidates)


def create_dataloader(
    processed_dir: str | Path = "data/processed",
    split: str = "train",
    delta: int = 4,
    image_size: int = 448,
    grid_size: int = 7,
    boxes_per_cell: int = 2,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    return_meta: bool = False,
    horizontal_flip: bool = False,
) -> DataLoader:
    """Build a DataLoader for MOT17 2D+t samples."""

    dataset = MOT17TrajectoryDataset(
        processed_dir=processed_dir,
        split=split,
        delta=delta,
        image_size=image_size,
        grid_size=grid_size,
        boxes_per_cell=boxes_per_cell,
        return_meta=return_meta,
        horizontal_flip=horizontal_flip,
    )
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def box_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def decode_target_boxes(
    target: torch.Tensor,
    motion_mask: torch.Tensor,
    image_size: int = 448,
    grid_size: int = 7,
    boxes_per_cell: int = 2,
    num_classes: int = 1,
    confidence_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Decode assigned YOLOv1 cells from the target tensor for visualization."""

    target = target.detach().cpu()
    motion_mask = motion_mask.detach().cpu()
    class_offset = boxes_per_cell * 5
    motion_offset = class_offset + num_classes
    boxes: list[dict[str, float]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            conf = float(target[row, col, 4])
            if conf < confidence_threshold:
                continue

            x_cell = float(target[row, col, 0])
            y_cell = float(target[row, col, 1])
            w = float(target[row, col, 2]) * image_size
            h = float(target[row, col, 3]) * image_size
            x = ((col + x_cell) / grid_size) * image_size
            y = ((row + y_cell) / grid_size) * image_size
            mx = float(target[row, col, motion_offset]) * image_size
            my = float(target[row, col, motion_offset + 1]) * image_size
            mw = float(target[row, col, motion_offset + 2]) * image_size
            mh = float(target[row, col, motion_offset + 3]) * image_size

            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "end_x": x + mx,
                    "end_y": y + my,
                    "end_w": w + mw,
                    "end_h": h + mh,
                    "motion_valid": float(motion_mask[row, col]),
                    "cell_row": float(row),
                    "cell_col": float(col),
                }
            )
    return boxes


def visualize_batch(
    batch: dict[str, Any] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_path: str | Path,
    image_size: int = 448,
    grid_size: int = 7,
    boxes_per_cell: int = 2,
    num_classes: int = 1,
    max_samples: int = 4,
) -> Path:
    """Save a visualization for one DataLoader batch on frame t.

    Each sample is drawn on a single image: the start box is shown on frame t,
    and an arrow from the box center indicates motion to t + delta.
    """

    if isinstance(batch, dict):
        images = batch["image"]
        targets = batch["target"]
        motion_masks = batch["motion_mask"]
    else:
        images, targets, motion_masks = batch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = min(max_samples, images.shape[0])
    canvas = Image.new("RGB", (image_size, image_size * samples), (35, 35, 35))

    for idx in range(samples):
        image = images[idx]
        frame_t = tensor_to_pil(image[:3])
        panel = frame_t.copy()
        draw = ImageDraw.Draw(panel)

        decoded = decode_target_boxes(
            targets[idx],
            motion_masks[idx],
            image_size=image_size,
            grid_size=grid_size,
            boxes_per_cell=boxes_per_cell,
            num_classes=num_classes,
        )
        for obj in decoded:
            start_box = box_xywh_to_xyxy(obj["x"], obj["y"], obj["w"], obj["h"])
            color = (44, 220, 110) if obj["motion_valid"] else (240, 170, 40)
            draw.rectangle(start_box, outline=color, width=2)
            draw_motion_arrow(draw, obj["x"], obj["y"], obj["end_x"], obj["end_y"], fill=(255, 230, 80))

        draw.text((8, 8), f"sample {idx}: frame t + motion", fill=(255, 255, 255))
        canvas.paste(panel, (0, idx * image_size))

    canvas.save(output_path)
    return output_path


def draw_motion_arrow(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)

    dx = x1 - x0
    dy = y1 - y0
    length = float((dx * dx + dy * dy) ** 0.5)
    if length < 1.0:
        return

    ux = dx / length
    uy = dy / length
    head_len = min(12.0, max(6.0, length * 0.25))
    head_w = head_len * 0.55
    base_x = x1 - ux * head_len
    base_y = y1 - uy * head_len
    left = (base_x - uy * head_w, base_y + ux * head_w)
    right = (base_x + uy * head_w, base_y - ux * head_w)
    draw.polygon([(x1, y1), left, right], fill=fill)


if __name__ == "__main__":
    config = CONFIG
    dataloader = create_dataloader(
        processed_dir=config["processed_dir"],
        split=str(config["split"]),
        delta=int(config["delta"]),
        image_size=int(config["image_size"]),
        grid_size=int(config["grid_size"]),
        boxes_per_cell=int(config["boxes_per_cell"]),
        batch_size=int(config["batch_size"]),
        shuffle=bool(config["shuffle"]),
        num_workers=int(config["num_workers"]),
        return_meta=True,
        horizontal_flip=bool(config["horizontal_flip"]),
    )
    batch = next(iter(dataloader))
    print(batch["image"].shape)
    print(batch["target"].shape)
    print(batch["motion_mask"].shape)
    print(batch["meta"])

    if bool(config["visualize"]):
        output_path = visualize_batch(
            batch,
            config["visualize_path"],
            image_size=int(config["image_size"]),
            grid_size=int(config["grid_size"]),
            boxes_per_cell=int(config["boxes_per_cell"]),
        )
        print(f"Saved visualization to {output_path}")
