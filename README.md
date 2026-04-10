# SANR-3D
**Selective and Noise-Resistant Virtual Point Cloud Fusion for Long-Range 3D Object Detection**

This repository provides the **modified files** for our SANR-3D framework, implemented **on top of the official VirConv codebase**.

Our method extends VirConv with three main components:

- **DNrC (Dilated Noise-Resistant Convolution)**: suppresses boundary-aligned artifacts in projected sparse features using 2D dilated sparse convolutions.
- **MCMA (Multiscale Cross Multi-Head Attention)**: performs geometry-consistent cross-modal fusion with **3D features as queries** and **2D features as keys/values**.
- **CasA (Cascade Attention Module)**: progressively refines RoI features and improves long-range localization stability.

In addition, we use **Stochastic Voxel Discard (StVD)**:

- **Input StVD**: distance-aware stochastic discard at the input stage.
- **Layer StVD**: random sparse voxel discard during training.

> **Important**
> This repository is released in a **patch-style form**.
> Please first clone the official **VirConv** repository, then copy the files from this repository into the corresponding locations and overwrite the original files.

---

## Table of Contents

- [Overview](#overview)
- [Modified Files](#modified-files)
- [Environment](#environment)
- [Installation](#installation)
- [How to Apply This Repository on Top of VirConv](#how-to-apply-this-repository-on-top-of-virconv)
- [Dataset Preparation](#dataset-preparation)
- [Optional: Generate Virtual Points with PENet](#optional-generate-virtual-points-with-penet)
- [Build](#build)
- [Training](#training)
- [Evaluation](#evaluation)
- [TensorBoard](#tensorboard)
- [Visualization](#visualization)
- [Selected Results](#selected-results)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Overview

Virtual point clouds can effectively densify sparse LiDAR observations at long range, but they also inherit depth estimation errors. In particular, inaccuracies near object boundaries and occlusions may produce **boundary-aligned artifacts** after back-projection, which can degrade downstream 3D detection if fused indiscriminately.

To address this issue, SANR-3D introduces a **selective and noise-resistant fusion framework**:

- **DNrC** enhances projected sparse features before cross-modal interaction.
- **MCMA** selectively injects informative image semantics into sparse 3D voxel features.
- **CasA** progressively refines detection results in a lightweight coarse-to-fine manner.

Our implementation follows the setting in which **depth completion is performed offline**, and the generated virtual points are used as **fixed early-fusion inputs** during detector training and evaluation.

---

## Modified Files

The core modifications are based on the original VirConv structure.

```text
VirConv/
├── tools/
│   ├── train.py
│   ├── test.py
│   └── cfgs/models/kitti/VirConv-L.yaml
├── pcdet/
│   ├── datasets/
│   │   ├── dataset.py
│   │   └── kitti/kitti_dataset_mm.py
│   └── models/
│       ├── backbones_3d/spconv_backbone.py
│       └── roi_heads/ted_head.py
└── tools/PENet/dataloaders/my_loader.py   # optional, only if you also release your PENet-side preprocessing
