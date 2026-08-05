# YOLO2D-t

Minimal YOLOv1-style `2D+t` training scaffold.

The repo keeps the original YOLOv1 training idea as the baseline:

- MSE loss for bbox coordinates
- MSE loss for object confidence
- MSE loss for no-object confidence
- MSE loss for class prediction
- one responsible box per occupied grid cell

The only research extension kept in this version is motion prediction.

## Input And Target

Expected dataloader output:

```python
{
    "image": Tensor[B, 6, 448, 448],
    "target": Tensor[B, 7, 7, 15],
    "motion_mask": Tensor[B, 7, 7],
}
```

The default setting assumes:

- input: 2 RGB frames concatenated along channel dimension
- grid size: `S = 7`
- boxes per cell: `B = 2`
- classes: `C = 1`
- motion output: `mx, my, mw, mh`

For each cell, the output layout is:

```text
box_1: x, y, w, h, confidence
box_2: x, y, w, h, confidence
class: person
motion: mx, my, mw, mh
```

So the final target shape is:

```text
7 x 7 x (2 * 5 + 1 + 4) = 7 x 7 x 15
```

## Setup

```bash
git clone https://github.com/thangSy221105/YOLO2D-t.git
cd YOLO2D-t
pip install -r requirements.txt
```

For Colab, use:

```text
notebooks/colab_train_yolo2dt.ipynb
```

## Train

```bash
python train.py --config configs/default.yaml
```

Resume latest checkpoint:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume latest
```

Resume a specific checkpoint and train until epoch 50:

```bash
python resume_yolo2dt.py --config configs/default.yaml --resume outputs/default/epoch_003.pt --epochs 50
```

## Motion Fine-Tune

After training the baseline, use motion fine-tuning when the detection branch is usable but motion prediction is still weak:

```bash
python finetune_motion_yolo2dt.py --config configs/default.yaml --checkpoint outputs/default/epoch_003.pt --epochs 10 --lr 1e-5
```

The fine-tune script keeps the same YOLOv1-style loss components, but changes the weights to focus more on motion:

```bash
python finetune_motion_yolo2dt.py \
  --config configs/default.yaml \
  --checkpoint outputs/default/epoch_003.pt \
  --epochs 10 \
  --lr 1e-5 \
  --lambda-coord 1.0 \
  --lambda-noobj 0.1 \
  --lambda-class 0.2 \
  --lambda-motion 5.0
```

Best motion checkpoint is saved as:

```text
outputs/motion_finetune/best_motion.pt
```

## YOLOv8 Joint Detect + Motion

The repo also includes a practical YOLOv8 branch that starts from a
detect-only checkpoint, upgrades the stem to 2-frame input, and predicts both
detection and motion on the existing `2D+t` grid targets.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run joint detect + motion fine-tuning from a YOLOv8 detect-only checkpoint:

```bash
python train_yolov8_motion.py \
  --config configs/yolov8_motion.yaml \
  --checkpoint /content/YOLO2D-t/outputs/yolov8_detect_only/weights/best.pt
```

Useful overrides:

```bash
python train_yolov8_motion.py \
  --config configs/yolov8_motion.yaml \
  --checkpoint /content/YOLO2D-t/outputs/yolov8_detect_only/weights/best.pt \
  --epochs 15 \
  --lr 5e-5 \
  --lambda-motion 5.0
```

The script prints validation benchmarks after each epoch by default:

```text
Precision, Recall, F1, AP@0.5, mean IoU
motion L1, center L2, future IoU
```

Benchmark frequency and thresholds can be changed:

```bash
python train_yolov8_motion.py \
  --config configs/yolov8_motion.yaml \
  --checkpoint yolov8n.pt \
  --benchmark-every 1 \
  --benchmark-conf 0.25 \
  --benchmark-iou 0.5
```

By default the YOLOv8 detector is frozen and only the upgraded 6-channel stem
plus the new joint heads are trained. If you want to let the detector adapt too:

```bash
python train_yolov8_motion.py \
  --config configs/yolov8_motion.yaml \
  --checkpoint /content/YOLO2D-t/outputs/yolov8_detect_only/weights/best.pt \
  --unfreeze-detector
```

## Notes

- This is not a full reproduction of the original YOLOv1 paper.
- The baseline intentionally avoids extra detector variants and non-YOLOv1 loss tricks.
- Motion loss is applied only where `motion_mask == 1`.
- The goal is to isolate one question: can a YOLOv1-style detector learn a short-term motion signal from multi-frame input?
