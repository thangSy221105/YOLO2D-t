# YOLO2D-t

Minimal training scaffold for a YOLOv1-style `2D+t` prototype.

This repo is designed for the case where the dataset already returns:

- `image`: `Tensor[B, 6, H, W]`
- `target`: `Tensor[B, S, S, B*5 + C + 4]`
- `motion_mask`: `Tensor[B, S, S]`

The default setup assumes:

- 2 RGB frames concatenated along channels: `6` input channels
- YOLOv1 grid: `S = 7`
- boxes per cell: `B = 2`
- classes: `C = 1`
- extra motion head: `mx, my, mw, mh`

## Repo structure

```text
.
├── configs/
│   └── default.yaml
├── src/
│   └── yolo2dt/
│       ├── __init__.py
│       ├── config.py
│       ├── data_adapter.py
│       ├── loss.py
│       ├── model.py
│       ├── trainer.py
│       └── utils.py
├── train.py
└── requirements.txt
```

## Colab setup

```bash
git clone https://github.com/thangSy221105/YOLO2D-t.git
cd YOLO2D-t
pip install -r requirements.txt
```

If your dataset code already exists in another file, the quickest path is:

1. Copy your dataset script into this repo.
2. Edit `src/yolo2dt/data_adapter.py`.
3. Replace the placeholder dataset builder with your real dataset/dataloader.

The scaffold already supports a direct import path for your earlier dataset file:

- expected module: `scripts.mot17_2dt_dataset`
- expected function: `create_dataloader(...)`

If that module exists, `train.py` will use it automatically.

## Training

```bash
python train.py --config configs/default.yaml
```

Resume the latest YOLOv1-style 2D+t checkpoint:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume latest
```

Resume a specific YOLOv1-style checkpoint and train until epoch 20:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume outputs/default/epoch_005.pt --epochs 20
```

The YOLOv1-style resume script also prints validation loss details, saves `best.pt`, reduces LR on plateau, and early-stops by default:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume latest --patience 5 --lr-patience 3
```

Resume with lower LR and focal confidence/no-object loss:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume outputs/default/epoch_003.pt --epochs 50 --lr 3e-5 --lambda-noobj 0.25 --use-focal-conf
```

Resume with auxiliary DIoU loss for bbox plateau:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume outputs/default/best.pt --epochs 50 --lr 1e-5 --lambda-noobj 0.25 --use-focal-conf --lambda-iou 1.0
```

For Colab, use `notebooks/colab_train_yolo2dt.ipynb`.

For the YOLOv8-style 2D+t prototype, use:

```bash
python train_yolov8_2dt.py --config configs/yolov8_2dt.yaml
python visualize_yolov8_2dt.py --config configs/yolov8_2dt.yaml
```

Resume the latest YOLOv8-style checkpoint:

```bash
python resume_yolov8_2dt.py --config configs/yolov8_2dt.yaml --resume latest
```

Resume a specific checkpoint and train until epoch 20:

```bash
python resume_yolov8_2dt.py --config configs/yolov8_2dt.yaml --resume outputs/yolov8_2dt/epoch_005.pt --epochs 20
```

Colab notebook:

```text
notebooks/colab_train_yolov8_2dt.ipynb
```

## Expected batch format

Each batch can be either:

```python
{
    "image": Tensor[B, 6, 448, 448],
    "target": Tensor[B, 7, 7, 15],
    "motion_mask": Tensor[B, 7, 7],
}
```

or:

```python
(image, target, motion_mask)
```

## Notes

- This is a baseline scaffold, not a reproduction of the original YOLOv1 paper.
- The loss assumes one GT object per occupied cell, consistent with standard YOLOv1 grid assignment.
- For cells with objects, the loss picks the responsible predicted box using IoU against the GT box.
- Motion loss is only applied where `motion_mask == 1`.

## Next steps

Once this trains and produces predictions, the next improvements are usually:

1. Replace the tiny backbone with a stronger one.
2. Improve the decode and evaluation pipeline.
3. Add validation metrics for detection and motion separately.
4. Add checkpoint resume and mixed precision tuning.
