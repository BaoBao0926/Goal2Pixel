import os
import torch
import numpy as np


def _to_numpy(x):
    """Move torch tensor to CPU and to numpy; accept numpy arrays as-is."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def _to_hwc_uint8(img):
    """Ensure HWC and uint8 for RGB saving. Accept float [0,1] or [0,255]."""
    img = _to_numpy(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # If likely 0..1 float -> scale; otherwise clip to 0..255
        if img.max() <= 1.0 + 1e-6:
            img = (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _parse_obj_id(s: str):
    """
    Accepts formats like '2_23_483' (level_region_object) or 'obj_483'.
    Returns (level_id or None, region_id or None, instance_id).
    """
    parts = re.split(r"[^\d]+", str(s))  # split on non-digits
    nums = [p for p in parts if p != ""]
    if len(nums) >= 3:  # e.g., ['2','23','483', ...]
        L, R, I = int(nums[0]), int(nums[1]), int(nums[2])
        return L, R, I
    elif len(nums) == 1:  # e.g., 'obj_483'
        return None, None, int(nums[0])
    else:
        raise ValueError(f"Unrecognized object id format: {s}")

def seg_idx_to_onehot(seg_idx: np.ndarray, num_classes: int) -> np.ndarray:
    """
    seg_idx: (H, W) int category ids in [0, num_classes-1]
    returns: (H, W, C) float32 one-hot
    """
    if seg_idx.ndim == 3 and seg_idx.shape[-1] == 1:
        seg_idx = seg_idx[..., 0]
    seg_idx = np.clip(seg_idx.astype(np.int32), 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.float32)[seg_idx]  # (H,W,C)
    return onehot

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


import pandas as pd
import re


def read_matterport_categories(category_file):
    with open(category_file, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    cols = re.split(r"\s+", header_line)  # e.g. ['index','raw_category','category',...,'mpcat40']


    # 2) Read the remaining lines as TAB-separated, using the header we just parsed
    df = pd.read_csv(
        category_file,
        sep=r"\s{2,}",  # split only on >=2 spaces
        header=None,
        names=cols,
        skiprows=1,
        engine="python"  # needed for regex separators
    )

    # Extract the "category" column as a list
    categories = df["category"].tolist()

    def read_last_tokens(path):
        last_values = []
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                # split on any whitespace (>=1 space or tab)
                parts = line.strip().split()
                if parts:
                    last_values.append(parts[-1])
                else:
                    last_values.append(None)
        return last_values

    mpcat40 = read_last_tokens(category_file)

    return categories


if __name__ == "__main__":
    category_file = "../../VLN_dataset/data/matterport_semantics/matterport_category_mappings.tsv"
    categories = read_matterport_categories(category_file)
    print(categories)