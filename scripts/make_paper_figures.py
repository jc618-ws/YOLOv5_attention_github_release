from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# 1. 路径与模型设置
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

RELEASE = ROOT
RESULTS = RELEASE / "results"
FIG = RESULTS / "paper_figures"

FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["Baseline", "A1", "A2", "ECA"]

METRICS_FILE = RESULTS / "final_comparison_seed0.csv"

print("=" * 65)
print("YOLOv5 四模型论文图表生成程序")
print("=" * 65)

# =====================================================
# 2. 检查训练记录
# =====================================================

print("\n========== 1. 检查四组训练记录 ==========")

training = {}

for model in MODELS:

    path = RESULTS / model / "results.csv"

    if not path.exists():
        raise FileNotFoundError(f"缺少训练记录: {path}")

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    required = [
        "epoch",
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{model}缺少字段: {missing}")

    if len(df) != 300:
        raise ValueError(
            f"{model}训练记录不是300轮，实际为{len(df)}"
        )

    training[model] = df

    print(f"{model}: {len(df)}轮，检查通过")

# =====================================================
# 3. 检查测试结果
# =====================================================

print("\n========== 2. 检查最终测试数据 ==========")

if not METRICS_FILE.exists():
    raise FileNotFoundError(METRICS_FILE)

table = pd.read_csv(METRICS_FILE, encoding="utf-8-sig")

if set(table["Model"]) != set(MODELS):
    raise ValueError(
        f"最终模型列表错误: {table['Model'].tolist()}"
    )

table = table.set_index("Model").loc[MODELS]

print(table.to_string())

# =====================================================
# 4. 检查测试曲线图片
# =====================================================

print("\n========== 3. 检查测试图表 ==========")

IMAGE_FILES = [
    "PR_curve.png",
    "F1_curve.png",
    "confusion_matrix.png"
]

for model in MODELS:

    folder = RESULTS / model / "test_results"

    for filename in IMAGE_FILES:

        path = folder / filename

        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"{model}缺少测试图片: {path}"
            )

    print(f"{model}: PR/F1/混淆矩阵全部正常")

# =====================================================
# 5. 保存图表函数
# =====================================================

def save_figure(fig, name):

    fig.savefig(
        FIG / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        FIG / f"{name}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"生成: {name}.png / {name}.pdf")


# =====================================================
# 6. 训练损失曲线
# =====================================================

print("\n========== 4. 绘制训练损失曲线 ==========")

loss_cols = [
    "train/box_loss",
    "train/obj_loss",
    "train/cls_loss"
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, loss_cols):

    for model in MODELS:

        df = training[model]

        ax.plot(
            df["epoch"],
            df[col],
            label=model,
            linewidth=1.6
        )

    ax.set_title(col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

fig.suptitle("Training Loss Comparison - Four Models")

save_figure(fig, "training_losses_4models")


# =====================================================
# 7. 训练mAP曲线
# =====================================================

print("\n========== 5. 绘制训练mAP曲线 ==========")

map_cols = [
    ("metrics/mAP_0.5", "mAP@0.5"),
    ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95")
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (col, title) in zip(axes, map_cols):

    for model in MODELS:

        df = training[model]

        ax.plot(
            df["epoch"],
            df[col],
            label=model,
            linewidth=1.6
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

fig.suptitle("Training mAP Comparison - Four Models")

save_figure(fig, "training_map_4models")


# =====================================================
# 8. 测试指标对比
# =====================================================

print("\n========== 6. 绘制测试指标对比图 ==========")

x = np.arange(len(MODELS))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(
    x - width / 2,
    table["mAP50"],
    width,
    label="mAP@0.5"
)

axes[0].bar(
    x + width / 2,
    table["mAP50_95"],
    width,
    label="mAP@0.5:0.95"
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(MODELS)
axes[0].set_ylim(0, 0.8)
axes[0].set_title("mAP Comparison")
axes[0].set_ylabel("Score")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(
    x - width / 2,
    table["Precision"],
    width,
    label="Precision"
)

axes[1].bar(
    x + width / 2,
    table["Recall"],
    width,
    label="Recall"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(MODELS)
axes[1].set_ylim(0, 0.9)
axes[1].set_title("Precision and Recall")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle("Test Metrics Comparison - Four Models")

save_figure(fig, "test_metrics_comparison_4models")


# =====================================================
# 9. PR、F1、混淆矩阵汇总
# =====================================================

print("\n========== 7. 绘制测试图表汇总 ==========")

for filename, title, output in [
    ("PR_curve.png", "PR Curve", "PR_curve_montage_4models"),
    ("F1_curve.png", "F1 Curve", "F1_curve_montage_4models"),
    (
        "confusion_matrix.png",
        "Confusion Matrix",
        "confusion_matrix_montage_4models"
    )
]:

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):

        path = RESULTS / model / "test_results" / filename

        img = plt.imread(path)

        ax.imshow(img)
        ax.set_title(model, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        title + " - Four Models",
        fontsize=16
    )

    save_figure(fig, output)


# =====================================================
# 10. 最终检查
# =====================================================

print("\n========== 8. 最终文件检查 ==========")

outputs = [
    "training_losses_4models",
    "training_map_4models",
    "test_metrics_comparison_4models",
    "PR_curve_montage_4models",
    "F1_curve_montage_4models",
    "confusion_matrix_montage_4models"
]

for name in outputs:

    for ext in [".png", ".pdf"]:

        path = FIG / (name + ext)

        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"生成失败: {path}")

        print(
            f"{path.name}: "
            f"{path.stat().st_size / 1024:.1f} KB"
        )

print("\n" + "=" * 65)
print("四模型论文图表全部生成成功")
print("模型: Baseline / A1 / A2 / ECA")
print("图表目录:", FIG)
print("=" * 65)
