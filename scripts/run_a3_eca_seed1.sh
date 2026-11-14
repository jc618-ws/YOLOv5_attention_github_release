#!/bin/bash
set -Eeuo pipefail

cd /root/CBAM-YOLOv5

CFG="models/yolov5s_v5_eca_voc.yaml"
DATA="results/four_model_archive_20261106_143407/Baseline/data.yaml"
HYP="runs/train/voc_v50_baseline_2501_seed1/hyp.yaml"
WEIGHTS="weights/pretrained/yolov5s_v5.0.pt"

NAME="voc_v50_a3_eca_2501_seed1"
RUN="runs/train/$NAME"

echo "===== A3 ECA 自动训练 ====="
date

# 1. 文件检查
for f in "$CFG" "$DATA" "$HYP" "$WEIGHTS" \
         train_voc_no_testval.py val.py; do
    test -f "$f" || {
        echo "文件不存在: $f"
        exit 1
    }
done

# 2. 防止覆盖已有实验
if [ -e "$RUN" ]; then
    echo "实验目录已存在，停止，避免覆盖数据: $RUN"
    exit 1
fi

# 3. 检查其他训练进程
if ps -eo pid,args | grep -E \
   '[p]ython .*train_voc_no_testval.py|[p]ython .*train.py'; then
    echo "发现其他训练进程，停止启动"
    exit 1
fi

# 4. 检查磁盘
df -h /

AVAIL=$(df -Pk . | awk 'NR==2 {print $4}')
if [ "$AVAIL" -lt 10485760 ]; then
    echo "可用磁盘空间不足10GB，停止训练"
    exit 1
fi

# 5. 检查GPU
nvidia-smi

# 6. 启动训练
echo "===== 开始 A3 训练 ====="

python train_voc_no_testval.py \
    --weights "$WEIGHTS" \
    --cfg "$CFG" \
    --data "$DATA" \
    --hyp "$HYP" \
    --epochs 300 \
    --batch-size 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --optimizer SGD \
    --patience 100 \
    --freeze 0 \
    --seed 1 \
    --project runs/train \
    --name "$NAME"

echo "===== A3训练完成 ====="

test -f "$RUN/weights/best.pt" || {
    echo "best.pt不存在，停止测试"
    exit 1
}

# 7. 自动测试独立测试集
echo "===== 开始独立测试集评估 ====="

python val.py \
    --data "$DATA" \
    --weights "$RUN/weights/best.pt" \
    --batch-size 16 \
    --imgsz 640 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --max-det 300 \
    --task test \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name "${NAME}_test"

echo "===== A3全部任务完成 ====="
date
