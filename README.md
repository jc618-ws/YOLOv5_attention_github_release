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
- `configs/VOC2007.yaml`: dataset configuration used by the training and evaluation scripts.
- `weights/<model>/best.pt`: final best checkpoint for Baseline, A1, A2, and ECA.
- `results/<model>/`: training CSV and test visualizations for each model.
- `results/paper_figures/`: figures used in the paper.
- `results/final_comparison_seed0.csv`: final common-test comparison table.

## Dataset

The converted VOC2007 YOLO-format dataset is distributed
separately as `VOC2007_YOLO_train_val_test.zip`.

Download the dataset archive from the Releases page:

https://github.com/jc618-ws/YOLOv5_attention_github_release/releases

Place the downloaded ZIP file in the repository root.

Extract the archive:

    unzip VOC2007_YOLO_train_val_test.zip -d .

Expected directory structure:

- datasets/VOC2007/YOLO/images/train
- datasets/VOC2007/YOLO/images/val
- datasets/VOC2007/YOLO/images/test
- datasets/VOC2007/YOLO/labels/train
- datasets/VOC2007/YOLO/labels/val
- datasets/VOC2007/YOLO/labels/test

The dataset configuration is `configs/VOC2007.yaml`.

## Training and evaluation

Activate a Python 3.10 environment with the required
dependencies and CUDA-enabled PyTorch.

### Training

Run the following commands from the repository root:

    bash scripts/train_baseline.sh
    bash scripts/train_a1.sh
    bash scripts/train_a2.sh
    bash scripts/train_eca.sh

Each script uses its corresponding model configuration
and hyperparameter file.

The experiments use 300 epochs, batch size 32,
640x640 input images, SGD, 8 workers, and seed 0.
Individual early-stopping and checkpoint-saving settings
are preserved in the training scripts.

### Evaluation

Run the following commands:

    bash scripts/test_baseline.sh
    bash scripts/test_a1.sh
    bash scripts/test_a2.sh
    bash scripts/test_eca.sh

All four models are evaluated on the common VOC2007
test split.

Evaluation settings:

- Image size: 640
- Batch size: 32
- Confidence threshold: 0.001
- NMS IoU threshold: 0.6
- Maximum detections: 300
- Test images: 4,952
- Object instances: 12,032

## Environment

The experiments were run on a Linux server with an NVIDIA GeForce RTX 4090 and Python 3.10.
Please install dependencies from `requirements.txt` and use a compatible CUDA-enabled PyTorch
build for your own hardware.

## Checkpoints

The `.pt` files are PyTorch checkpoints. Only load checkpoints from sources you trust.
SHA-256 hashes for the four final checkpoints are recorded in `SHA256SUMS`.

## Upstream code and acknowledgements

### YOLOv5

This project is based on the YOLOv5 codebase by Ultralytics.
The upstream license and notices are retained in this repository.

Source:
https://github.com/ultralytics/yolov5

### CBAM

The project implements and integrates CBAM attention modules
into YOLOv5. The implementation is located in
`models/attention.py` and `models/common.py`.

The repository also includes the original PyTorch BAM/CBAM
reference implementation by Jongchan Park in
`references/attention-module-master/`.

The original MIT License and copyright notice are retained
in that directory.

Reference:
Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S.
CBAM: Convolutional Block Attention Module.
European Conference on Computer Vision (ECCV), 2018.

### ECA

The ECA module is implemented in `models/attention.py`
and integrated into YOLOv5 through `models/common.py`.

Its implementation follows the Efficient Channel Attention
method described in the original ECA-Net paper.

Reference:
Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., and Hu, Q.
ECA-Net: Efficient Channel Attention for Deep Convolutional
Neural Networks.
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2020.

### BAM reference

The bundled BAM/CBAM reference repository also contains
the original BAM implementation.

Reference:
Park, J., Woo, S., Lee, J.-Y., and Kweon, I. S.
BAM: Bottleneck Attention Module.
British Machine Vision Conference (BMVC), 2018.

## Dataset Attribution and Copyright

This project uses the PASCAL Visual Object Classes
Challenge 2007 (VOC2007) dataset.

Original dataset:
https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/

Citation:

Everingham, M., Van Gool, L., Williams, C. K. I.,
Winn, J., and Zisserman, A.
The PASCAL Visual Object Classes Challenge 2007
(VOC2007) Results.

The images are sourced from the original VOC2007
dataset and remain subject to their respective
copyrights and original licenses.

This project converts the original VOC annotations
into YOLO format.

The project authors do not claim ownership of the
original images.

Redistribution of images is subject to the
applicable licenses and permissions of their
respective rights holders.
