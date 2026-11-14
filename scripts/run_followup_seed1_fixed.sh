#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/CBAM-YOLOv5

TAG=$(date +%Y%m%d_%H%M%S)
LOG="logs/followup_seed1_fixed_${TAG}.log"

mkdir -p logs
echo "$LOG" > logs/followup_seed1_fixed.latest
exec > >(tee -a "$LOG") 2>&1

echo "===== 自动化实验启动 ====="
echo "时间：$(date)"
echo "日志：$LOG"

# 防止重复启动
exec 9>/tmp/cbam_followup_seed1_fixed.lock
if ! flock -n 9; then
    echo "已有自动化流程运行，停止重复启动"
    exit 1
fi

ARCHIVE=results/four_model_archive_20261106_143407
BACKUP=backups/paper_cbam_20261106_140702

BASE_RUN=runs/train/voc_v50_baseline_2501_seed1
A1_RUN=runs/train/voc_v50_a1_nores_2501_seed1

DATA="$ARCHIVE/Baseline/data.yaml"
HYP="$ARCHIVE/Baseline/hyp.yaml"
PRETRAIN=weights/pretrained/yolov5s_v5.0.pt

# 优先使用归档的A1配置
if [ -f "$ARCHIVE/A1/model.yaml" ]; then
    A1_CFG="$ARCHIVE/A1/model.yaml"
else
    A1_CFG=models/yolov5s_v5_cbam_nores_voc.yaml
fi

echo "===== 1. 检查文件 ====="

for f in "$DATA" "$HYP" "$PRETRAIN" "$A1_CFG"; do
    if [ ! -f "$f" ]; then
        echo "缺少文件：$f"
        exit 1
    fi
done

# 确保模型源码与之前备份一致
for f in common.py yolo.py attention.py; do
    diff -q "models/$f" "$BACKUP/$f"
done

echo "模型源码检查通过"
echo "A1配置：$A1_CFG"

# Baseline必须已经完整训练
if [ ! -s "$BASE_RUN/results.csv" ] ||
   [ "$(wc -l < "$BASE_RUN/results.csv")" -ne 301 ]; then
    echo "Baseline Seed1训练记录不完整，停止"
    exit 1
fi

if [ ! -s "$BASE_RUN/weights/best.pt" ]; then
    echo "Baseline best.pt不存在"
    exit 1
fi

# 避免覆盖已有A1实验
if [ -e "$A1_RUN" ]; then
    echo "A1 Seed1实验目录已经存在，停止以避免覆盖"
    exit 1
fi

# 只检查真正的训练进程，不检查脚本自身或普通shell
TRAIN_PIDS=$(ps -eo pid=,comm=,args= |
    awk '$2 ~ /^python/ && /train_voc_no_testval[.]py/ {print $1}')

if [ -n "$TRAIN_PIDS" ]; then
    echo "发现其他训练进程：$TRAIN_PIDS"
    exit 1
fi

GPU_PIDS=$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader 2>/dev/null | awk 'NF {print $1}')

if [ -n "$GPU_PIDS" ]; then
    echo "GPU仍有计算进程：$GPU_PIDS"
    exit 1
fi

python -c "import torch; assert torch.cuda.is_available(); print('CUDA正常:', torch.cuda.get_device_name(0))"

# 2. Baseline Seed1测试集评估
echo "===== 2. Baseline Seed1测试集评估 ====="

python val.py \
    --weights "$BASE_RUN/weights/best.pt" \
    --data "$DATA" \
    --task test \
    --img 640 \
    --batch-size 16 \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name "voc_v50_baseline_2501_seed1_test_${TAG}"

echo "Baseline Seed1测试集评估完成"

# 3. A1 Seed1训练
echo "===== 3. A1 Seed1开始训练 ====="

python train_voc_no_testval.py \
    --weights "$PRETRAIN" \
    --cfg "$A1_CFG" \
    --data "$DATA" \
    --hyp "$HYP" \
    --epochs 300 \
    --batch-size 32 \
    --img 640 \
    --device 0 \
    --workers 8 \
    --seed 1 \
    --project runs/train \
    --name voc_v50_a1_nores_2501_seed1

echo "A1训练命令正常结束"

if [ ! -s "$A1_RUN/weights/best.pt" ]; then
    echo "A1 best.pt不存在，停止"
    exit 1
fi

echo "A1实际完成轮数：$(($(wc -l < "$A1_RUN/results.csv") - 1))"

# 4. A1测试集评估
echo "===== 4. A1 Seed1测试集评估 ====="

python val.py \
    --weights "$A1_RUN/weights/best.pt" \
    --data "$DATA" \
    --task test \
    --img 640 \
    --batch-size 16 \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name "voc_v50_a1_nores_2501_seed1_test_${TAG}"

# 5. 完成
echo "===== 所有任务完成 ====="
echo "Baseline Seed1：$BASE_RUN"
echo "A1 Seed1：$A1_RUN"
echo "日志：$LOG"
echo "完成时间：$(date)"

