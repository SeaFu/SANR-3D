# Selective and Noise-Resistant Virtual Point Cloud Fusion for Long-Range 3D Object Detection

This repository provides the official code release for our paper:

**Selective and Noise-Resistant Virtual Point Cloud Fusion for Long-Range 3D Object Detection**

Our method builds upon the VirConv framework and introduces a selective and noise-resistant multimodal fusion design for long-range 3D object detection.

The proposed framework includes three main components:

- **DNrC (Dilated Noise-Resistant Convolution)**: suppresses boundary-aligned artifacts in projected sparse features using 2D dilated sparse convolutions.
- **MCMA (Multiscale Cross Multi-Head Attention)**: performs geometry-consistent cross-modal fusion with **3D features as queries** and **2D features as keys/values**.
- **CasA (Cascade Attention Module)**: progressively refines RoI features and improves long-range localization stability.

In addition, we use **Stochastic Voxel Discard (StVD)**:

- **Input StVD**: distance-aware stochastic discard at the input stage.
- **Layer StVD**: random sparse voxel discard during training.

> **Important**  
> This repository is a **standalone code release** of our implementation.  
> Users can clone this repository directly, install the dependencies, prepare the dataset, and then run training or evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Main Implementation Changes](#main-implementation-changes)
- [Environment](#environment)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Optional: Generate Virtual Points with PENet](#optional-generate-virtual-points-with-penet)
- [Build](#build)
- [Pretrained Model](#pretrained-model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Distance-wise Evaluation](#distance-wise-evaluation)
- [TensorBoard](#tensorboard)
- [Selected Results](#selected-results)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Overview

Virtual point clouds can effectively densify sparse LiDAR observations at long range, but they also inherit depth estimation errors. In particular, inaccuracies near object boundaries and occlusions may produce **boundary-aligned artifacts** after back-projection, which can degrade downstream 3D detection if virtual points are fused indiscriminately.

To address this issue, our method introduces a **selective and noise-resistant fusion framework**:

- **DNrC** enhances projected sparse features before cross-modal interaction.
- **MCMA** selectively injects informative image semantics into sparse 3D voxel features.
- **CasA** progressively refines detection results in a lightweight coarse-to-fine manner.

Our implementation follows the setting in which **depth completion is performed offline**, and the generated virtual points are used as **fixed early-fusion inputs** during detector training and evaluation.

---

## Repository Structure

```text
SANR-3D/
├── pcdet/
├── tools/
│   ├── PENet/
│   ├── cfgs/
│   ├── eval_utils/
│   ├── train_utils/
│   ├── train.py
│   ├── test.py
│   └── eval_by_distance.py
├── requirements.txt
└── setup.py
```

---

## Main Implementation Changes

The core files corresponding to our method are:

- **`pcdet/datasets/dataset.py`**
  - Implements **input-stage stochastic voxel discard (Input StVD)** with distance-aware sampling.

- **`pcdet/datasets/kitti/kitti_dataset_mm.py`**
  - Provides **KITTI multimodal dataset loading** and multimodal info generation for fused LiDAR and virtual point cloud inputs.

- **`pcdet/models/backbones_3d/spconv_backbone.py`**
  - Implements the backbone modifications, including:
    - **DNrC (Dilated Noise-Resistant Convolution)**
    - **MCMA (Multiscale Cross Multi-Head Attention)**
    - **layer-level stochastic voxel discard (Layer StVD)**

- **`pcdet/models/roi_heads/ted_head.py`**
  - Implements the detection head with cascaded RoI refinement.

- **`tools/train.py`, `tools/test.py`**
  - Training and evaluation entry points used in our experiments.

- **`tools/cfgs/models/kitti/VirConv-L.yaml`**
  - Main experiment configuration used in our paper.

- **`tools/PENet/dataloaders/my_loader.py`**
  - Optional preprocessing utilities for the PENet-based virtual point generation pipeline.

---

## Environment

The following environment is recommended:

- Ubuntu 20.04
- Python 3.9
- CUDA 11.1
- PyTorch 1.8.1 + cu111
- spconv 2.1.22

---

## Installation

Create a clean conda environment:

```bash
conda create -n virconv-cu111 python=3.9 -y
conda activate virconv-cu111
```

Install PyTorch:

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

Install the main dependencies:

```bash
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 \
    waymo-open-dataset-tf-2-5-0 nuscenes-devkit==1.0.5 \
    spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely \
    matplotlib opencv-python addict pyquaternion awscli open3d \
    pandas future pybind11 tensorboardX tensorboard Cython \
    prefetch-generator
```

Then install the project in development mode:

```bash
python setup.py develop
```

---

## Dataset Preparation

Please prepare the **official KITTI 3D Object Detection** dataset in the following layout:

```text
SANR-3D/
├── data/
│   ├── kitti/
│   │   ├── ImageSets/
│   │   ├── training/
│   │   │   ├── calib/
│   │   │   ├── velodyne/
│   │   │   ├── label_2/
│   │   │   ├── image_2/
│   │   │   ├── planes/            # optional
│   │   │   └── velodyne_depth/    # required for multimodal training/evaluation
│   │   ├── testing/
│   │   │   ├── calib/
│   │   │   ├── velodyne/
│   │   │   ├── image_2/
│   │   │   └── velodyne_depth/
│   │   ├── gt_database_mm/
│   │   ├── kitti_dbinfos_train_mm.pkl
│   │   ├── kitti_infos_train.pkl
│   │   ├── kitti_infos_val.pkl
│   │   ├── kitti_infos_trainval.pkl
│   │   └── kitti_infos_test.pkl
```

Generate KITTI info files with:

```bash
python3 -m pcdet.datasets.kitti.kitti_dataset_mm create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
```

---

## Optional: Generate Virtual Points with PENet

If you want to regenerate the virtual point clouds, you can use the included PENet pipeline.

Example:

```bash
cd tools/PENet
python3 main.py --detpath ../../data/kitti/training
python3 main.py --detpath ../../data/kitti/testing
```

This should generate the `velodyne_depth` folders used by the multimodal detector.

> In our setting, dense depth maps are generated **offline** and converted into virtual points before detector training.

---

## Build

After installing the dependencies, build the project in development mode:

```bash
python setup.py develop
```

---

## Pretrained Model

We also provide a pretrained checkpoint for convenience:

- **Google Drive**: [final_checkpoint.pth](https://drive.google.com/file/d/1kkjwbcSF3mUjdyKTuQam4hZDz2oLtFUC/view?usp=sharing)

### Example usage

Place the downloaded checkpoint under:

```text
output/models/kitti/VirConv-L/default/ckpt/
```

Then evaluate it with:

```bash
cd tools
python3 test.py --cfg_file cfgs/models/kitti/VirConv-L.yaml \
    --batch_size 4 \
    --ckpt ../output/models/kitti/VirConv-L/default/ckpt/final_checkpoint.pth \
    --save_to_file
```

> If you store the checkpoint in another directory, simply replace the `--ckpt` path accordingly.

---

## Training

### Single-GPU training

Adjust `BATCH_SIZE_PER_GPU` in `tools/cfgs/models/kitti/VirConv-L.yaml` according to your available GPU memory.

```bash
cd tools
python3 train.py --cfg_file cfgs/models/kitti/VirConv-L.yaml
```

### Multi-GPU training

You may use the provided distributed training scripts if needed:

```bash
cd tools
bash dist_train.sh
```

---

## Evaluation

### Evaluate a checkpoint

```bash
cd tools
python3 test.py --cfg_file cfgs/models/kitti/VirConv-L.yaml \
    --batch_size 20 \
    --ckpt ../output/models/kitti/VirConv-L/default/ckpt/checkpoint_epoch_66.pth
```

### Evaluate and save prediction files

```bash
cd tools
python3 test.py --cfg_file cfgs/models/kitti/VirConv-L.yaml \
    --batch_size 20 \
    --ckpt ../output/models/kitti/VirConv-L/default/ckpt/checkpoint_epoch_66.pth \
    --save_to_file
```

---

## Distance-wise Evaluation

You may also evaluate distance-binned performance with:

```bash
cd tools
python3 eval_by_distance.py
```

Please edit the dataset and prediction paths inside the script if needed.

---

## TensorBoard

```bash
# Training logs
tensorboard --logdir=./output/models/kitti/VirConv-L/default/tensorboard/

# Evaluation logs
tensorboard --logdir=./output/models/kitti/VirConv-L/default/eval/eval_with_train/tensorboard_val
```

---

## Selected Results

> KITTI Val / Car / AP@0.7 / R40

### Overall improvement over VirConv

| Metric     | Easy  | Moderate | Hard  |
|------------|-------|----------|-------|
| **3D AP**  | +1.96 | +1.40    | +2.16 |
| **BEV AP** | +0.04 | +1.94    | +0.94 |

### Distance-binned improvement (Moderate / 3D AP)

| Range   | 0-15 m | 15-30 m | 30-45 m | 45 m+ |
|---------|--------|---------|---------|-------|
| Gain    | +2.25  | +2.57   | +5.64   | +2.70 |

> In our experiments, inserting MCMA at Stage 3/4 provided the best accuracy-efficiency trade-off. DNrC improved robustness to boundary-induced noise, while cascaded RoI refinement further strengthened long-range localization with modest overhead.

---

## FAQ

### Why is the depth completion network not trained jointly with the detector?

Our setting uses **offline depth completion**. Dense depth maps are generated first and then converted into virtual points before detector training. Therefore, gradients are **not** propagated back to the depth completion model.

### Why use stochastic voxel discard?

Dense virtual points can significantly increase computation, especially in near-range regions. StVD reduces redundant computation while preserving useful long-range structure.

### Why can far-range AP sometimes be unstable?

At extreme distance, valid detections may sometimes be penalized by incomplete annotations. Qualitative inspection is often helpful for interpreting such cases.

---

## Acknowledgements

This project is built on top of the following excellent works:

- VirConv
- OpenPCDet
- TED
- CasA
- PENet
- SFD

We sincerely thank the authors for making their code and research available.

---

## Citation

If you find this project useful, please cite both the upstream VirConv work and our paper.

### VirConv

```bibtex
@inproceedings{VirConv,
  title={Virtual Sparse Convolution for Multimodal 3D Object Detection},
  author={Wu, Hai and Wen, Chenglu and Shi, Shaoshuai and Wang, Cheng},
  booktitle={CVPR},
  year={2023}
}
```

### Ours

```bibtex
@article{SANR3D,
  title={Selective and Noise-Resistant Virtual Point Cloud Fusion for Long-Range 3D Object Detection},
  author={Hsu, Chih-Ming and Fu, Yuan-Ting and Wang, Bo-Yun},
  journal={Under review},
  year={2026}
}
```

> Please replace the BibTeX entry above with the final published version once available.

---

## License

This project is released under the **Apache-2.0 License**, consistent with the upstream **VirConv** repository.  
Please also retain the original copyright notice and license terms from VirConv when redistributing or modifying the upstream code.
