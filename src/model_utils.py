"""
Model loading and inference for the RF (pixel-based) and CNN (patch-based)
land cover classifiers.
"""

from __future__ import annotations

import joblib
import numpy as np
import torch

from .architectures import SimpleCNN

PATCH_SIZE = 16
N_INPUT_CHANNELS = 15  # 10 raw bands + 5 spectral indices
CLASSES = ["water", "urban", "vegetation", "sand", "clouds"]
N_CLASSES = len(CLASSES)


def load_cnn(path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load the CNN. Handles three cases transparently, since torch.load can
    hand back different things depending on how the file was saved:
      1. A full nn.Module object (torch.save(model, path))
      2. A checkpoint dict with a nested state_dict, e.g.
         {"model_state_dict": ..., "epoch": ..., ...}
      3. A raw state_dict (torch.save(model.state_dict(), path))
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # It's a dict: either a raw state_dict or a checkpoint wrapping one.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # assume it's already a raw state_dict

        model = SimpleCNN(
            in_channels=N_INPUT_CHANNELS,
            num_classes=N_CLASSES,
            patch_size=PATCH_SIZE,
        )
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model


_RF_COMMON_KEYS = ["model", "rf_model", "rf", "classifier", "estimator", "clf"]


def load_rf(path: str):
    """
    Load the scikit-learn RandomForest model saved with joblib.

    Handles both a bare model (joblib.dump(rf, path)) and a dict wrapping
    it under a common key (e.g. joblib.dump({"model": rf, ...}, path)).
    """
    obj = joblib.load(path)

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for key in _RF_COMMON_KEYS:
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]
        raise ValueError(
            f"'{path}' loaded as a dict but no model with a .predict() method "
            f"was found under the usual keys {_RF_COMMON_KEYS}. "
            f"Found keys: {list(obj.keys())}. Tell me which key holds the "
            f"actual model and I'll update load_rf to use it."
        )

    raise ValueError(
        f"'{path}' loaded as a {type(obj)}, which has no .predict() method "
        f"and isn't a dict either. Please check how this file was saved."
    )


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    """
    Starting positions for sliding a `patch`-sized window across a
    dimension of length `size`, moving by `stride` each step. Always
    includes a final window flush with the edge so the whole image is
    covered, even if that means the last window overlaps the previous one
    more than `stride`.
    """
    if size <= patch:
        return [0]
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] != size - patch:
        positions.append(size - patch)
    return positions


@torch.no_grad()
def classify_cnn_patchwise(
    model: torch.nn.Module,
    img: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Slide a `patch_size` window across a standardized (C, H, W) image with
    step `stride`, classify each patch with the CNN (one label per patch),
    and paint that label across the whole patch region.

    If `stride < patch_size`, patches overlap: each pixel's final class is
    the majority vote across every patch that covers it. If `stride is
    None`, defaults to non-overlapping tiling (stride == patch_size).

    Returns:
        class_map: (H, W) int array of predicted class indices.
    """
    if stride is None:
        stride = patch_size

    c, h, w = img.shape
    row_starts = _window_starts(h, patch_size, stride)
    col_starts = _window_starts(w, patch_size, stride)

    patches = []
    coords = []
    for y0 in row_starts:
        for x0 in col_starts:
            patch = img[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            patches.append(patch)
            coords.append((y0, x0))

    patches_tensor = torch.stack(patches).to(device)  # (N, C, ph, pw)

    preds = []
    for i in range(0, patches_tensor.shape[0], batch_size):
        batch = patches_tensor[i : i + batch_size]
        logits = model(batch)
        batch_preds = logits.argmax(dim=1).cpu().numpy()
        preds.append(batch_preds)
    preds = np.concatenate(preds)

    # Accumulate per-pixel class votes so overlapping patches are resolved
    # by majority vote rather than the last patch simply overwriting others.
    votes = np.zeros((h, w, N_CLASSES), dtype=np.int32)
    for (y0, x0), pred in zip(coords, preds):
        votes[y0 : y0 + patch_size, x0 : x0 + patch_size, pred] += 1

    return votes.argmax(axis=2).astype(np.int64)


def classify_rf_pixelwise(rf_model, img: torch.Tensor) -> np.ndarray:
    """
    Classify a standardized (C, H, W) image pixel-by-pixel with the RF.

    The RF was trained with 1-indexed class labels (1..N), while CLASSES
    is 0-indexed (0..N-1) to match the CNN's argmax output and the color
    map. We remap predictions using rf_model.classes_ -- the sorted array
    of label values the RF actually saw during training -- rather than
    naively subtracting 1, so this stays correct even if the raw labels
    aren't a clean contiguous 1..N range.

    Returns:
        class_map: (H, W) int array of predicted class indices, 0-indexed
        to match CLASSES.
    """
    c, h, w = img.shape
    features = img.permute(1, 2, 0).reshape(-1, c).numpy()  # (H*W, C)
    preds = rf_model.predict(features)

    # rf_model.classes_ is sorted ascending, e.g. [1, 2, 3, 4, 5]. Position
    # of a label within that array is exactly its 0-indexed class index,
    # assuming training used labels in the same order as CLASSES.
    if hasattr(rf_model, "classes_"):
        preds = np.searchsorted(rf_model.classes_, preds)
    else:
        # Fallback: assume a simple 1-indexed -> 0-indexed shift.
        preds = preds - 1

    return preds.reshape(h, w).astype(np.int64)