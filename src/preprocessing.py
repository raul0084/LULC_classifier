from __future__ import annotations

import numpy as np
import torch

BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
N_RAW_BANDS = len(BAND_ORDER)
N_INDICES = 5
N_TOTAL_CHANNELS = N_RAW_BANDS + N_INDICES  # 15

EPS = 1e-6


def add_bands(img_tensor: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Compute spectral indices and append them, then standardize every channel.

    This is a direct port of the training-time function -- same band
    indices, same index formulas, same per-band standardization.

    Args:
        img_tensor: float tensor of shape (10, H, W), bands in BAND_ORDER.

    Returns:
        img_tensor: standardized tensor of shape (15, H, W)
        n_bands: number of channels (15)
        means: per-channel means used for standardization, shape (15,)
        stds: per-channel stds used for standardization, shape (15,)
    """
    if img_tensor.shape[0] != N_RAW_BANDS:
        raise ValueError(
            f"Expected {N_RAW_BANDS} raw bands, got {img_tensor.shape[0]}. "
            f"Expected band order: {BAND_ORDER}"
        )

    img_tensor = img_tensor.clone().float()

    blue = img_tensor[0]
    green = img_tensor[1]
    red = img_tensor[2]
    nir = img_tensor[7]
    swir1 = img_tensor[8]
    swir2 = img_tensor[9]

    ndvi = (nir - red) / (nir + red + EPS)
    mndwi = (green - swir1) / (green + swir1 + EPS)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)
    mbi = (swir1 - swir2 - nir) / (swir1 + swir2 + nir + EPS)
    ndbi = (swir1 - nir) / (swir1 + nir + EPS)

    indices = torch.stack([ndvi, mndwi, bsi, mbi, ndbi])  # (5, H, W)
    img_tensor = torch.cat([img_tensor, indices], dim=0)  # (15, H, W)

    means = []
    stds = []
    for b in range(img_tensor.shape[0]):
        mean = img_tensor[b].mean()
        std = img_tensor[b].std()
        means.append(mean)
        stds.append(std)
        img_tensor[b] = (img_tensor[b] - mean) / (std + EPS)

    means = torch.stack(means)
    stds = torch.stack(stds)
    n_bands = img_tensor.shape[0]

    return img_tensor, n_bands, means, stds


def stack_to_tensor(bands_array: np.ndarray) -> torch.Tensor:
    """
    Convert a (10, H, W) numpy array (as read from rasterio) into a float
    torch tensor ready for add_bands().
    """
    if bands_array.shape[0] != N_RAW_BANDS:
        raise ValueError(
            f"Expected {N_RAW_BANDS} bands in raster, got {bands_array.shape[0]}. "
            f"Expected band order: {BAND_ORDER}. If your .tif has bands in a "
            f"different order, reorder them before calling this function."
        )
    return torch.from_numpy(bands_array.astype(np.float32))
