"""Preprocess MOT17 into short-clip indices for YOLOv1 2D+t.

The script keeps only one MOT17 detector variant (FRCNN by default), filters
ground-truth pedestrians, splits by sequence, and writes lightweight CSV files.
Images are not copied; the dataset reads the original MOT17 frames.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from statistics import mean


DEFAULT_TRAIN_BASES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"]
DEFAULT_VAL_BASES = ["MOT17-11", "MOT17-13"]

CONFIG = {
    "mot17_root": Path("data/MOT17/train"),
    "output_dir": Path("data/processed"),
    "detector": "FRCNN",
    "delta": 20,
    "visibility_threshold": 0.25,
    "grid_size": 7,
    "image_size": 448,
    "train_bases": DEFAULT_TRAIN_BASES,
    "val_bases": DEFAULT_VAL_BASES,
}


def read_seqinfo(seq_dir: Path) -> dict[str, int | str]:
    config = ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    section = config["Sequence"]
    return {
        "name": section.get("name", seq_dir.name),
        "im_dir": section.get("imDir", "img1"),
        "seq_length": section.getint("seqLength"),
        "im_width": section.getint("imWidth"),
        "im_height": section.getint("imHeight"),
        "im_ext": section.get("imExt", ".jpg"),
    }


def letterbox_params(width: int, height: int, image_size: int) -> tuple[float, float, float]:
    scale = min(image_size / width, image_size / height)
    pad_x = (image_size - width * scale) / 2.0
    pad_y = (image_size - height * scale) / 2.0
    return scale, pad_x, pad_y


def read_annotations(
    seq_dir: Path,
    sequence_base: str,
    visibility_threshold: float,
) -> list[dict[str, int | float | str]]:
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")

    rows: list[dict[str, int | float | str]] = []
    with gt_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            left = float(row[2])
            top = float(row[3])
            width = float(row[4])
            height = float(row[5])
            mark = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])

            if mark != 1 or class_id != 1 or visibility < visibility_threshold:
                continue
            if width <= 0 or height <= 0:
                continue

            rows.append(
                {
                    "sequence": sequence_base,
                    "frame": frame,
                    "track_id": track_id,
                    "xc": left + width / 2.0,
                    "yc": top + height / 2.0,
                    "w": width,
                    "h": height,
                    "visibility": visibility,
                    "class_id": 0,
                }
            )
    return rows


def choose_sequence_dirs(mot17_root: Path, detector: str) -> dict[str, Path]:
    seq_dirs: dict[str, Path] = {}
    for seq_dir in sorted(mot17_root.glob(f"*-{detector}")):
        base = "-".join(seq_dir.name.split("-")[:2])
        seq_dirs[base] = seq_dir
    return seq_dirs


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sequence_list(path: Path, sequences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sequences) + "\n", encoding="utf-8")


def build_pairs(
    sequence_base: str,
    seq_dir: Path,
    seqinfo: dict[str, int | str],
    annotations: list[dict[str, int | float | str]],
    delta: int,
    project_root: Path,
) -> tuple[list[dict], dict[int, dict[int, dict]]]:
    by_frame_track: dict[int, dict[int, dict]] = defaultdict(dict)
    for ann in annotations:
        by_frame_track[int(ann["frame"])][int(ann["track_id"])] = ann

    img_dir = str(seqinfo["im_dir"])
    im_ext = str(seqinfo["im_ext"])
    seq_length = int(seqinfo["seq_length"])
    rows: list[dict] = []
    for frame_t in range(1, seq_length - delta + 1):
        frame_next = frame_t + delta
        img_t = seq_dir / img_dir / f"{frame_t:06d}{im_ext}"
        img_next = seq_dir / img_dir / f"{frame_next:06d}{im_ext}"
        if not img_t.exists() or not img_next.exists():
            continue

        tracks_t = by_frame_track.get(frame_t, {})
        tracks_next = by_frame_track.get(frame_next, {})
        motion_valid = len(set(tracks_t) & set(tracks_next))
        rows.append(
            {
                "sequence": sequence_base,
                "seq_dir": seq_dir.name,
                "frame_t": frame_t,
                "frame_next": frame_next,
                "img_t": img_t.relative_to(project_root).as_posix(),
                "img_next": img_next.relative_to(project_root).as_posix(),
                "num_objects_t": len(tracks_t),
                "num_motion_valid": motion_valid,
            }
        )
    return rows, by_frame_track


def collision_stats(
    pair_rows: list[dict],
    by_frame_track: dict[int, dict[int, dict]],
    seqinfo: dict[str, int | str],
    image_size: int,
    grid_size: int,
) -> dict[str, float | int]:
    scale, pad_x, pad_y = letterbox_params(
        int(seqinfo["im_width"]),
        int(seqinfo["im_height"]),
        image_size,
    )

    frames_with_collision = 0
    total_objects = 0
    lost_objects = 0
    objects_per_frame: list[int] = []

    for pair in pair_rows:
        frame_t = int(pair["frame_t"])
        cell_counts: dict[tuple[int, int], int] = defaultdict(int)
        anns = by_frame_track.get(frame_t, {}).values()
        objects_per_frame.append(len(list(anns)))

        for ann in by_frame_track.get(frame_t, {}).values():
            x = float(ann["xc"]) * scale + pad_x
            y = float(ann["yc"]) * scale + pad_y
            col = min(grid_size - 1, max(0, int((x / image_size) * grid_size)))
            row = min(grid_size - 1, max(0, int((y / image_size) * grid_size)))
            cell_counts[(row, col)] += 1
            total_objects += 1

        collisions = sum(max(0, count - 1) for count in cell_counts.values())
        if collisions:
            frames_with_collision += 1
            lost_objects += collisions

    return {
        "frames_with_grid_collision": frames_with_collision,
        "collision_frame_ratio": frames_with_collision / len(pair_rows) if pair_rows else 0.0,
        "objects_dropped_if_one_object_per_cell": lost_objects,
        "object_drop_ratio": lost_objects / total_objects if total_objects else 0.0,
        "avg_objects_per_frame_t": mean(objects_per_frame) if objects_per_frame else 0.0,
    }


def main() -> None:
    config = CONFIG
    delta = int(config["delta"])
    if delta <= 0:
        raise ValueError("CONFIG['delta'] must be positive")

    mot17_root = Path(config["mot17_root"])
    output_dir = Path(config["output_dir"])
    project_root = output_dir.resolve().parent.parent
    detector = str(config["detector"])
    visibility_threshold = float(config["visibility_threshold"])
    grid_size = int(config["grid_size"])
    image_size = int(config["image_size"])
    train_bases = list(config["train_bases"])
    val_bases = list(config["val_bases"])

    seq_dirs = choose_sequence_dirs(mot17_root, detector)
    requested_bases = sorted(set(train_bases + val_bases))
    missing = [base for base in requested_bases if base not in seq_dirs]
    if missing:
        raise FileNotFoundError(
            f"Missing {detector} sequences under {mot17_root}: {', '.join(missing)}"
        )

    annotations_dir = output_dir / "annotations"
    pairs_dir = output_dir / "pairs"
    splits_dir = output_dir / "splits"
    stats_dir = output_dir / "stats"

    split_map = {"train": train_bases, "val": val_bases}
    all_stats: dict[str, object] = {
        "detector": detector,
        "delta": delta,
        "visibility_threshold": visibility_threshold,
        "image_size": image_size,
        "grid_size": grid_size,
        "splits": {},
        "sequences": {},
    }

    pair_rows_by_split: dict[str, list[dict]] = {"train": [], "val": []}
    for split, bases in split_map.items():
        write_sequence_list(splits_dir / f"{split}_sequences.txt", bases)

        for base in bases:
            seq_dir = seq_dirs[base]
            seqinfo = read_seqinfo(seq_dir)
            annotations = read_annotations(seq_dir, base, visibility_threshold)
            write_rows(
                annotations_dir / f"{base}.csv",
                ["sequence", "frame", "track_id", "xc", "yc", "w", "h", "visibility", "class_id"],
                annotations,
            )

            pair_rows, by_frame_track = build_pairs(
                base,
                seq_dir,
                seqinfo,
                annotations,
                delta,
                project_root,
            )
            pair_rows_by_split[split].extend(pair_rows)

            stats = collision_stats(pair_rows, by_frame_track, seqinfo, image_size, grid_size)
            stats.update(
                {
                    "seq_dir": seq_dir.name,
                    "seq_length": int(seqinfo["seq_length"]),
                    "width": int(seqinfo["im_width"]),
                    "height": int(seqinfo["im_height"]),
                    "annotations": len(annotations),
                    "pairs": len(pair_rows),
                    "pairs_with_motion": sum(1 for row in pair_rows if int(row["num_motion_valid"]) > 0),
                }
            )
            all_stats["sequences"][base] = stats

    for split, rows in pair_rows_by_split.items():
        write_rows(
            pairs_dir / f"{split}_delta{delta}.csv",
            [
                "sequence",
                "seq_dir",
                "frame_t",
                "frame_next",
                "img_t",
                "img_next",
                "num_objects_t",
                "num_motion_valid",
            ],
            rows,
        )
        all_stats["splits"][split] = {
            "sequences": split_map[split],
            "pairs": len(rows),
            "pairs_with_motion": sum(1 for row in rows if int(row["num_motion_valid"]) > 0),
        }

    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "dataset_statistics.json").write_text(
        json.dumps(all_stats, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote processed dataset to {output_dir}")


if __name__ == "__main__":
    main()
