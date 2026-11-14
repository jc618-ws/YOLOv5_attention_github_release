#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/CBAM-YOLOv5

# ==========================================
# 实验配置
# ==========================================

BASE_PID=32574

BASE_RUN="voc_v50_baseline_2501_seed1"
BASE_DIR="runs/train/${BASE_RUN}"
BASE_LOG="logs/baseline_seed1_train.log"

ARCHIVE="results/four_model_archive_20261106_143407/Baseline"

WEIGHTS="weights/pretrained/yolov5s_v5.0.pt"
DATA="${ARCHIVE}/data.yaml"
HYP="${BASE_DIR}/hyp.yaml"

A1_CFG="models/yolov5s_v5_cbam_nores_voc.yaml"
A1_RUN="voc_v50_a1_nores_2501_seed1"
A1_DIR="runs/train/${A1_RUN}"

BASE_TEST="voc_v50_baseline_2501_seed1_test"
A1_TEST="voc_v50_a1_nores_2501_seed1_test"

BASE_TEST_LOG="logs/baseline_seed1_test.log"
A1_TRAIN_LOG="logs/a1_nores_seed1_train.log"
A1_TEST_LOG="logs/a1_nores_seed1_test.log"

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    log "错误：$*"
    exit 1
}

trap 'rc=$?; log "脚本异常退出，状态码=$rc，行号=${LINENO}"' ERR

# ==========================================
# 1. 环境和配置检查
# ==========================================

log "开始自动化实验流程"

for f in \
    "$WEIGHTS" \
    "$DATA" \
    "$HYP" \
    "$A1_CFG" \
    "$BASE_DIR/results.csv" \
    "$BASE_LOG"
do
    [[ -f "$f" ]] || die "文件不存在：$f"
done

cmp -s \
    models/yolov5s_v5_voc.yaml \
    "${ARCHIVE}/model.yaml" \
    || die "Baseline模型配置发生变化"

cmp -s \
    datasets/VOC2007/YOLO/VOC_A2.yaml \
    "$DATA" \
    || die "数据集配置发生变化"

cmp -s \
    runs/train/voc_v50_baseline_2501/hyp.yaml \
    "$HYP" \
    || die "训练超参数不一致"

grep -q 'C3_CBAM_NoRes' "$A1_CFG" \
    || die "A1配置中没有找到C3_CBAM_NoRes"

grep -q 'class C3_CBAM_NoRes' models/common.py \
    || die "A1模块源码不存在"

grep -q 'C3_CBAM_NoRes' models/yolo.py \
    || die "模型解析器没有识别A1模块"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA不可用'
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
"

# 防止覆盖已有实验

for d in \
    "runs/val/${BASE_TEST}" \
    "$A1_DIR" \
    "runs/val/${A1_TEST}"
do
    [[ ! -e "$d" ]] || die "实验目录已经存在，拒绝覆盖：$d"
done

log "配置检查通过"

# ==========================================
# 2. 等待 Baseline Seed1 完成
# ==========================================

if [[ -r "/proc/${BASE_PID}/cmdline" ]]; then

    CMD=$(tr '\0' ' ' < "/proc/${BASE_PID}/cmdline")

    [[ "$CMD" == *"train_voc_no_testval.py"* ]] \
        || die "PID ${BASE_PID} 不是预期训练进程"

    [[ "$CMD" == *"$BASE_RUN"* ]] \
        || die "PID ${BASE_PID} 对应的实验不正确"

    log "等待 Baseline Seed1 完成，PID=${BASE_PID}"

    while [[ -r "/proc/${BASE_PID}/cmdline" ]]; do

        STATE=$(ps -p "$BASE_PID" -o stat= 2>/dev/null || true)

        [[ "$STATE" == Z* ]] && break

        CMD=$(tr '\0' ' ' < "/proc/${BASE_PID}/cmdline" 2>/dev/null || true)

        [[ -z "$CMD" ]] && break

        [[ "$CMD" == *"$BASE_RUN"* ]] \
            || die "PID发生变化或进程异常"

        EPOCHS=$(($(wc -l < "$BASE_DIR/results.csv") - 1))

        log "Baseline Seed1 正在训练：${EPOCHS}/300"

        sleep 60
    done

else
    log "Baseline PID已经退出，检查训练结果"
fi

# ==========================================
# 3. 验证 Baseline 训练完成
# ==========================================

[[ -s "$BASE_DIR/weights/best.pt" ]] \
    || die "Baseline best.pt不存在"

[[ -s "$BASE_DIR/weights/last.pt" ]] \
    || die "Baseline last.pt不存在"

LINES=$(wc -l < "$BASE_DIR/results.csv")

(( LINES >= 2 )) \
    || die "Baseline训练记录不完整"

grep -q 'Results saved to' "$BASE_LOG" \
    || die "Baseline训练日志没有正常结束标记"

OTHER=$(ps -eo pid=,stat=,args= |
    awk '$2 !~ /^Z/ && /train_voc_no_testval.py/ {print $1}')

[[ -z "$OTHER" ]] \
    || die "发现其他训练进程，停止后续操作：$OTHER"

log "Baseline Seed1训练完成"
log "实际记录轮数：$((LINES - 1))"

# ==========================================
# 4. Baseline Seed1测试集评估
# ==========================================

log "开始 Baseline Seed1 测试集评估"

python val.py \
    --weights "$BASE_DIR/weights/best.pt" \
    --data "$DATA" \
    --task test \
    --img 640 \
    --batch-size 16 \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name "$BASE_TEST" \
    > "$BASE_TEST_LOG" 2>&1

grep -q 'Results saved to' "$BASE_TEST_LOG" \
    || die "Baseline测试集评估没有正常结束"

log "Baseline Seed1测试完成"

# ==========================================
# 5. 自动训练 A1 Seed1
# ==========================================

log "开始 A1 Seed1 训练"

python train_voc_no_testval.py \
    --weights "$WEIGHTS" \
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
    --name "$A1_RUN" \
    > "$A1_TRAIN_LOG" 2>&1

[[ -s "$A1_DIR/weights/best.pt" ]] \
    || die "A1 best.pt不存在"

[[ -s "$A1_DIR/weights/last.pt" ]] \
    || die "A1 last.pt不存在"

[[ -s "$A1_DIR/results.csv" ]] \
    || die "A1训练记录不存在"

grep -q 'Results saved to' "$A1_TRAIN_LOG" \
    || die "A1训练没有正常结束"

log "A1 Seed1训练完成"

# ==========================================
# 6. A1 Seed1测试集评估
# ==========================================

log "开始 A1 Seed1 测试集评估"

python val.py \
    --weights "$A1_DIR/weights/best.pt" \
    --data "$DATA" \
    --task test \
    --img 640 \
    --batch-size 16 \
    --device 0 \
    --workers 8 \
    --project runs/val \
    --name "$A1_TEST" \
    > "$A1_TEST_LOG" 2>&1

grep -q 'Results saved to' "$A1_TEST_LOG" \
    || die "A1测试集评估没有正常结束"

# ==========================================
# 7. 流程完成
# ==========================================

log "===================================="
log "全部自动化实验完成"
log "Baseline Seed1: $BASE_DIR"
log "Baseline Test: runs/val/$BASE_TEST"
log "A1 Seed1: $A1_DIR"
log "A1 Test: runs/val/$A1_TEST"
log "===================================="

