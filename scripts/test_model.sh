#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
  baseline) MODEL="Baseline" ;;
  a1)       MODEL="A1" ;;
  a2)       MODEL="A2" ;;
  eca)      MODEL="ECA" ;;
  *)
    echo "Usage: bash scripts/test_model.sh baseline|a1|a2|eca"
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"
WEIGHT="$ROOT/weights/$MODEL/best.pt"

# 自动定位数据集
if [[ -n "${VOC_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$VOC_DATA_ROOT"
elif [[ -d "$ROOT/datasets/VOC2007/YOLO/images/test" ]]; then
    DATA_ROOT="$ROOT/datasets/VOC2007/YOLO"
elif [[ -d "$ROOT/../datasets/VOC2007/YOLO/images/test" ]]; then
    DATA_ROOT="$ROOT/../datasets/VOC2007/YOLO"
else
    echo "错误：找不到VOC2007数据集"
    echo "请设置 VOC_DATA_ROOT 环境变量"
    exit 1
fi

if [[ ! -f "$WEIGHT" ]]; then
    echo "权重不存在: $WEIGHT"
    exit 1
fi

if [[ ! -f "$ROOT/configs/VOC2007.yaml" ]]; then
    echo "数据配置不存在"
    exit 1
fi

echo "========== 测试准备 =========="
echo "模型: $MODEL"
echo "权重: $WEIGHT"
echo "数据集: $DATA_ROOT"

# 检查测试集数量及标签
"$PYTHON_BIN" - "$DATA_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

images = {
    p.stem for p in (root / "images/test").iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

labels = {
    p.stem for p in (root / "labels/test").glob("*.txt")
}

instances = sum(
    sum(1 for line in p.read_text().splitlines() if line.strip())
    for p in (root / "labels/test").glob("*.txt")
)

print("测试图片:", len(images))
print("测试标签:", len(labels))
print("目标实例:", instances)

if len(images) != 4952:
    raise SystemExit("测试图片数量不正确")

if images != labels:
    raise SystemExit("图片和标签文件不匹配")

if instances != 12032:
    raise SystemExit("目标实例数量不正确")

print("数据集检查通过")
PY

# 创建临时数据配置，不修改正式配置
TMP_DATA="$(mktemp /tmp/voc2007-test-XXXXXX.yaml)"
trap 'rm -f "$TMP_DATA"' EXIT

"$PYTHON_BIN" - \
  "$ROOT/configs/VOC2007.yaml" \
  "$DATA_ROOT" \
  "$TMP_DATA" <<'PY'
import yaml
import sys
from pathlib import Path

source, data_root, output = sys.argv[1:]

with open(source, encoding="utf-8") as f:
    data = yaml.safe_load(f)

data["path"] = str(Path(data_root).resolve())

with open(output, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("临时测试配置创建成功")
PY

echo ""
echo "========== 开始测试 =========="

"$PYTHON_BIN" "$ROOT/val.py" \
  --data "$TMP_DATA" \
  --weights "$WEIGHT" \
  --batch-size 32 \
  --imgsz 640 \
  --conf-thres 0.001 \
  --iou-thres 0.6 \
  --max-det 300 \
  --task test \
  --device "${YOLO_DEVICE:-0}" \
  --workers 8 \
  --project "$ROOT/runs/test_reproduction" \
  --name "$MODEL" \
  --verbose
