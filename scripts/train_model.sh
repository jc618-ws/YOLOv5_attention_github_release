#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
  baseline)
    MODEL="Baseline"
    RUN_NAME="voc_v50_baseline_2501"
    PATIENCE=100
    SAVE_PERIOD=-1
    ;;
  a1)
    MODEL="A1"
    RUN_NAME="voc_v50_a1_nores_2501"
    PATIENCE=100
    SAVE_PERIOD=-1
    ;;
  a2)
    MODEL="A2"
    RUN_NAME="voc_v50_a2_multiscale_cbam_trainval"
    PATIENCE=300
    SAVE_PERIOD=50
    ;;
  eca)
    MODEL="ECA"
    RUN_NAME="voc_v50_a3_eca_2501_seed0_w8_control"
    PATIENCE=100
    SAVE_PERIOD=-1
    ;;
  *)
    echo "Usage: bash scripts/train_model.sh baseline|a1|a2|eca"
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"

echo "========== YOLOv5 Training =========="
echo "Model: $MODEL"
echo "Run: $RUN_NAME"
echo "Patience: $PATIENCE"
echo "Save period: $SAVE_PERIOD"

for F in \
  "$ROOT/configs/$MODEL/model.yaml" \
  "$ROOT/configs/$MODEL/hyp.yaml" \
  "$ROOT/configs/VOC2007.yaml" \
  "$ROOT/weights/pretrained/yolov5s_v5.0.pt"
do
  if [[ ! -f "$F" ]]; then
    echo "Missing file: $F"
    exit 1
  fi
done

for F in \
  images/train images/val images/test \
  labels/train labels/val labels/test
do
  if [[ ! -d "$ROOT/datasets/VOC2007/YOLO/$F" ]]; then
    echo "Dataset directory missing: $F"
    echo "Please download and extract VOC2007_YOLO_train_val_test.zip"
    exit 1
  fi
done

"$PYTHON_BIN" -c "import torch, numpy, yaml; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available())"

exec "$PYTHON_BIN" train_voc_no_testval.py \
  --weights "$ROOT/weights/pretrained/yolov5s_v5.0.pt" \
  --cfg "$ROOT/configs/$MODEL/model.yaml" \
  --data "$ROOT/configs/VOC2007.yaml" \
  --hyp "$ROOT/configs/$MODEL/hyp.yaml" \
  --epochs 300 \
  --batch-size 32 \
  --imgsz 640 \
  --optimizer SGD \
  --workers 8 \
  --device "${YOLO_DEVICE:-0}" \
  --project "$ROOT/runs/train" \
  --name "$RUN_NAME" \
  --patience "$PATIENCE" \
  --freeze 0 \
  --save-period "$SAVE_PERIOD" \
  --seed 0
