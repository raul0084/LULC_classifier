from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .model_utils import CLASSES

# Fixed, intuitive colors per class (order must match CLASSES)
CLASS_COLORS = {
    "water": "#2166ac",
    "urban": "#b2182b",
    "vegetation": "#1a9850",
    "sand": "#d9c06b",
    "clouds": "#e0e0e0",
}
COLOR_LIST = [CLASS_COLORS[c] for c in CLASSES]
CMAP = ListedColormap(COLOR_LIST)


def plot_class_map(class_map: np.ndarray, title: str):
    """Return a matplotlib Figure showing a single classified map."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(class_map, cmap=CMAP, vmin=0, vmax=len(CLASSES) - 1, interpolation="none")
    ax.set_title(title)
    ax.axis("off")
    return fig


def legend_figure():
    """Return a small standalone legend figure shared by both maps."""
    fig, ax = plt.subplots(figsize=(5, 0.6))
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES]
    ax.legend(handles=handles, loc="center", ncol=len(CLASSES), frameon=False)
    ax.axis("off")
    return fig


def agreement_map(map_a: np.ndarray, map_b: np.ndarray):
    """Return a figure highlighting where the two models agree/disagree."""
    agree = (map_a == map_b).astype(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(agree, cmap=ListedColormap(["#b2182b", "#1a9850"]), vmin=0, vmax=1)
    ax.set_title(f"Agreement: {agree.mean() * 100:.1f}%")
    ax.axis("off")
    return fig
