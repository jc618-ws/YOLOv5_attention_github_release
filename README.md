# YOLOv5 Attention Reproduction on PASCAL VOC2007

This repository contains the source code, model configurations, final trained checkpoints,
training records, and test-result visualizations for the paper. The final paper compares
**Baseline, A1, A2, and ECA**.

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
