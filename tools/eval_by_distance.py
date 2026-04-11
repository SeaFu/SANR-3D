#!/usr/bin/env python3
"""eval_by_distance_fixed.py  (v4)
---------------------------------
Compute **distance-wise 3D AP (Car, moderate, R11 or R40)** on the KITTI
*validation* split, and report the number of *moderate-Car GT* boxes in each distance bin.

Key features
~~~~~~~~~~~~
* Robust masking – every ndarray field keeps consistent length.
* Suppresses Numba GPU occupancy / parallel warnings.
* Optional `--metric R11|R40` (default: R40).

Example
~~~~~~~
```bash
python eval_by_distance_fixed.py \
  --kitti_info /path/to/kitti_infos_val.pkl \
  --det_pkl   /path/to/result.pkl \
  --bins 0 15 30 45 inf           \
  --metric R11                    # or R40
```
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numba.core.errors import NumbaPerformanceWarning  # type: ignore
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

# ----------------------------------------------------------------------
#  Silence annoying GPU‑occupancy / parallel diagnostics from Numba
# ----------------------------------------------------------------------
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Grid size .* will likely result in GPU under-utilization",
)
warnings.filterwarnings(
    "ignore",
    message=r"The keyword argument 'parallel=True' was specified but no transformation",
)

# ----------------------------------------------------------------------
#  Helper functions
# ----------------------------------------------------------------------

def l2_xy_distance(boxes_lidar: np.ndarray) -> np.ndarray:
    """Return Euclidean distance √(x² + y²) of each box centre in LiDAR coords."""
    return np.linalg.norm(boxes_lidar[:, :2], axis=1)


def moderate_car_mask(ann: dict) -> np.ndarray:
    """Boolean mask of *moderate‑difficulty Car* boxes inside **one** GT anno dict."""
    names = ann["name"]
    is_car = names == "Car"
    if "difficulty" in ann:
        is_mod = ann["difficulty"] == 1  # 0=easy,1=mod,2=hard  by OpenPCDet create_data
    else:
        # Fallback: compute via KITTI rules (bbox height>=25, occl<=2, trunc<=0.3)
        h = ann["bbox"][:, 3] - ann["bbox"][:, 1]
        is_mod = (h >= 25) & (ann["occluded"] <= 2) & (ann["truncated"] <= 0.3)
    return is_car & is_mod


def _copy_with_mask(src_ann: dict, mask: np.ndarray) -> dict:
    """Shallow‑copy anno dict, applying *mask* only to ndarray fields of same length."""
    new_ann = {}
    for k, v in src_ann.items():
        if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]:
            new_ann[k] = v[mask]
        else:
            new_ann[k] = v
    return new_ann


def split_annos_by_dist(
    gt_annos: List[dict],
    dt_annos: List[dict],
    bins: List[float],
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Split GT & detection annos into distance bins defined by *bins*."""
    n_bins = len(bins) - 1
    gt_splits = [[] for _ in range(n_bins)]
    dt_splits = [[] for _ in range(n_bins)]

    for g_ann, d_ann in zip(gt_annos, dt_annos):
        g_dist = l2_xy_distance(g_ann["gt_boxes_lidar"])
        d_dist = (
            l2_xy_distance(d_ann["boxes_lidar"])
            if len(d_ann) and "boxes_lidar" in d_ann
            else np.empty((0,))
        )

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            g_mask = (g_dist >= lo) & (g_dist < hi)
            d_mask = (d_dist >= lo) & (d_dist < hi)

            gt_splits[i].append(_copy_with_mask(g_ann, g_mask))
            dt_splits[i].append(_copy_with_mask(d_ann, d_mask))
    return gt_splits, dt_splits


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distance‑wise 3D‑AP evaluator for KITTI (Car, moderate)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kitti_info", required=True, type=Path, help="kitti_infos_val.pkl")
    p.add_argument("--det_pkl", required=True, type=Path, help="Detection result pkl")
    p.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default=[0, 15, 30, 45, float("inf")],
        help="Bin edges in metres (last one can be 'inf')",
    )
    p.add_argument(
        "--metric",
        choices=["R11", "R40"],
        default="R40",
        help="Recall sampling metric (R11 = 11‑point, R40 = 40‑point)",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load data
    with args.kitti_info.open("rb") as f:
        kitti_infos = pickle.load(f)
    gt_annos = [info["annos"] for info in kitti_infos]
    dt_annos = pickle.load(args.det_pkl.open("rb"))

    bins = args.bins
    if not np.isinf(bins[-1]):
        bins.append(float("inf"))

    gt_splits, dt_splits = split_annos_by_dist(gt_annos, dt_annos, bins)

    metric_key_suffix = f"moderate_{args.metric}"
    title_metric = args.metric
    print(f"\nDistance-bin 3D-AP (Car, moderate, {title_metric}) + #GT:")

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        hi_txt = f"{hi:.0f}" if np.isfinite(hi) else "inf"

        # Count moderate-Car GT boxes in this bin
        mod_gt_cnt = 0
        for g in gt_splits[i]:
            mod_gt_cnt += moderate_car_mask(g).sum()

        # Compute AP
        try:
            _, ap_dict = kitti_eval.get_official_eval_result(
                gt_splits[i], dt_splits[i], current_classes=[0]
            )
            ap = ap_dict[f"Car_3d/{metric_key_suffix}"]
            print(f"{lo:4.0f}–{hi_txt:<4} m : {ap:6.2f} %")
        except Exception:
            print(f"{lo:4.0f}–{hi_txt:<4} m :   n/a        (GT={mod_gt_cnt})")


if __name__ == "__main__":
    main()
