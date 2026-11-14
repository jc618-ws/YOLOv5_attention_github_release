#!/usr/bin/env python3
"""Prepare a clean GitHub release folder for the final YOLOv5 paper (no A0).

Run on the training server:
    python prepare_yolov5_github_release.py

Expected project root: /root/CBAM-YOLOv5
Expected model archive: results/four_model_archive_20261106_143407
"""
from pathlib import Path
import hashlib
import shutil
import tarfile
import textwrap

ROOT = Path('/root/CBAM-YOLOv5').resolve()
ARCH = ROOT / 'results/four_model_archive_20261106_143407'
OUT = ROOT / 'YOLOv5_attention_github_release'
MODELS = ('Baseline', 'A1', 'A2', 'ECA')

if not ROOT.is_dir():
    raise SystemExit(f'Project directory not found: {ROOT}')
if not ARCH.is_dir():
    raise SystemExit(f'Model archive not found: {ARCH}')
missing = [m for m in MODELS if not (ARCH / m / 'best.pt').is_file()]
if missing:
    raise SystemExit('Missing best.pt for: ' + ', '.join(missing))

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# Copy the source tree, excluding datasets, training outputs, archived results,
# checkpoints, caches, and local IDE/environment files.
EXCLUDE_DIRS = {
    '.git', '.idea', '.vscode', '.pytest_cache', '.mypy_cache', '__pycache__',
    '.venv', 'venv', 'env', 'datasets', 'runs', 'results', 'logs', 'weights',
    'wandb', 'tensorboard', 'YOLOv5_attention_github_release',
}
EXCLUDE_SUFFIXES = {'.pyc', '.cache'}

def ignore(directory, names):
    ignored = []
    for name in names:
        p = Path(directory) / name
        if name in EXCLUDE_DIRS or p.suffix in EXCLUDE_SUFFIXES:
            ignored.append(name)
        elif name.startswith('yolov5s_v5_a0'):
            ignored.append(name)
    return ignored

for item in ROOT.iterdir():
    if item.name in EXCLUDE_DIRS or item.name.startswith('yolov5s_v5_a0'):
        continue
    target = OUT / item.name
    if item.is_dir():
        shutil.copytree(item, target, ignore=ignore)
    elif item.is_file() and item.suffix not in EXCLUDE_SUFFIXES:
        shutil.copy2(item, target)

# Copy the four exact final checkpoints and their recorded model/hyperparameter configs.
for model in MODELS:
    src = ARCH / model
    dst = OUT / 'weights' / model
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / 'best.pt', dst / 'best.pt')

    cfg_dst = OUT / 'configs' / model
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for fname in ('model.yaml', 'hyp.yaml'):
        if (src / fname).is_file():
            shutil.copy2(src / fname, cfg_dst / fname)

    result_dst = OUT / 'results' / model
    result_dst.mkdir(parents=True, exist_ok=True)
    if (src / 'results.csv').is_file():
        shutil.copy2(src / 'results.csv', result_dst / 'results.csv')
    test_dir = src / 'test_results_common'
    if not test_dir.is_dir():
        test_dir = src / 'test_results'
    if test_dir.is_dir():
        shutil.copytree(test_dir, result_dst / 'test_results', dirs_exist_ok=True)

# Add the final common-test table and paper figures.
comparison = ARCH / 'final_comparison_seed0.csv'
if comparison.is_file():
    (OUT / 'results').mkdir(exist_ok=True)
    shutil.copy2(comparison, OUT / 'results' / comparison.name)
figures = ARCH / 'figures'
if figures.is_dir():
    shutil.copytree(figures, OUT / 'results' / 'paper_figures', dirs_exist_ok=True)

# Provide a dataset-path template; VOC images/labels are intentionally not redistributed.
(OUT / 'configs').mkdir(exist_ok=True)
(OUT / 'configs' / 'VOC2007_template.yaml').write_text(textwrap.dedent('''\
    # Edit path to the local YOLO-format VOC2007 directory before training/testing.
    path: /PATH/TO/VOC2007/YOLO
    train: images/train
    val: images/val
    test: images/test
    nc: 20
    names: [aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
            diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
            train, tvmonitor]
'''), encoding='utf-8')

# Optional initial pretrained checkpoint, if present in the original project.
pretrained = ROOT / 'weights' / 'pretrained' / 'yolov5s_v5.0.pt'
if pretrained.is_file():
    p = OUT / 'weights' / 'pretrained'
    p.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pretrained, p / pretrained.name)

# A clear README, based only on the recorded final comparison table.
readme = '''# YOLOv5 Attention Reproduction on PASCAL VOC2007

This repository contains the source code, model configurations, final trained checkpoints,
training records, and test-result visualizations for the paper. The final paper compares
**Baseline, A1, A2, and ECA**. The former A0 experiment is intentionally not included.

## Models and common test results

The table below is transcribed from `results/final_comparison_seed0.csv`.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Baseline | 0.736 | 0.611 | 0.669 | 0.441 |
| A1 (backbone CBAM, no outer residual) | 0.770 | 0.604 | 0.681 | 0.449 |
| A2 (neck multi-scale CBAM) | 0.762 | 0.605 | 0.677 | 0.447 |
| ECA (backbone ECA) | 0.748 | 0.623 | 0.684 | 0.452 |

The common test split contains 4,952 images and 12,032 object instances.
The recorded training setup uses 640x640 images, 300 epochs, batch size 32, and SGD.
Per-run options are preserved in the archived training records where available; consult
`results/` and each model's configuration before reproducing an individual run.

## Repository layout

- `models/`, `utils/`, `train.py`, `val.py`, and related files: YOLOv5 source code and project changes.
- `configs/<model>/model.yaml`: exact model configuration copied from the experiment archive.
- `configs/<model>/hyp.yaml`: training hyperparameters copied from the experiment archive.
- `configs/VOC2007_template.yaml`: edit the dataset path for your local VOC2007 YOLO-format dataset.
- `weights/<model>/best.pt`: final best checkpoint for Baseline, A1, A2, and ECA.
- `results/<model>/`: training CSV and test visualizations for each model.
- `results/paper_figures/`: figures used in the paper.
- `results/final_comparison_seed0.csv`: final common-test comparison table.

## Dataset

The PASCAL VOC2007 dataset is not included. Download it from its official distribution
and convert/arrange it in YOLO format. Edit `configs/VOC2007_template.yaml` so `path`
points to the resulting directory. The expected layout is `images/{train,val,test}` and
`labels/{train,val,test}`.

## Training and evaluation

Use the training and validation scripts included in this repository. First inspect the
available arguments with `python train_voc_no_testval.py --help` and `python val.py --help`.
Set the `--cfg` argument to the corresponding `configs/<model>/model.yaml`, the data
argument to `configs/VOC2007_template.yaml`, and the hyperparameter argument to the
corresponding `configs/<model>/hyp.yaml`. Use the final checkpoint under `weights/<model>/best.pt`
for evaluation. The exact command-line options can vary with the YOLOv5 script version.

## Environment

The experiments were run on a Linux server with an NVIDIA GeForce RTX 4090 and Python 3.10.
Please install dependencies from `requirements.txt` and use a compatible CUDA-enabled PyTorch
build for your own hardware.

## Checkpoints

The `.pt` files are PyTorch checkpoints. Only load checkpoints from sources you trust.
SHA-256 hashes for the four final checkpoints are recorded in `SHA256SUMS`.

## Upstream code and acknowledgements

This project is based on the YOLOv5 codebase by Ultralytics and includes project-specific
changes for CBAM/ECA experiments. Preserve the upstream license and notices included in
the source tree. Please cite the original YOLOv5, CBAM, ECA-Net, and PASCAL VOC publications
when using this repository.
'''
(OUT / 'README.md').write_text(readme, encoding='utf-8')

# Ignore local datasets, caches, and future training outputs.
(OUT / '.gitignore').write_text('''\
__pycache__/
*.py[cod]
*.cache
.DS_Store
.idea/
.vscode/
datasets/
runs/
logs/
wandb/
''', encoding='utf-8')

# Weight integrity manifest.
lines = []
for model in MODELS:
    p = OUT / 'weights' / model / 'best.pt'
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(f'{h}  weights/{model}/best.pt')
(OUT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# A compact archive for transfer; the directory remains available for git init/push.
archive_path = ROOT / 'YOLOv5_attention_github_release.tar.gz'
if archive_path.exists():
    archive_path.unlink()
with tarfile.open(archive_path, 'w:gz') as tf:
    tf.add(OUT, arcname=OUT.name)

print('GitHub release folder created:', OUT)
print('Archive created:', archive_path)
print('Included models:', ', '.join(MODELS))
print('A0 excluded. Dataset, runs, logs, and local caches excluded.')
print('Checkpoint SHA256:')
print((OUT / 'SHA256SUMS').read_text())
