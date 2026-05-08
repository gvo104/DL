from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "font.size": 10,
        }
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_environment(output_dir: Path, cache_dir: Path, figures_dir: Path, data_dir: Path) -> torch.device:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    return get_device()


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 160, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def flatten_images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim == 3:
        return x.reshape(len(x), -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Unsupported image array shape: {x.shape}")


def plot_class_distribution(
    train_counts: Sequence[int],
    test_counts: Sequence[int],
    *,
    save_path: Path,
    title: str = "Распределение классов MNIST",
    dpi: int = 160,
    show: bool = False,
) -> None:
    labels = np.arange(10)
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars1 = ax.bar(labels - width / 2, train_counts, width=width, label="train")
    bars2 = ax.bar(labels + width / 2, test_counts, width=width, label="test")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(train_counts) * 0.005,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(labels)
    ax.set_xlabel("Цифра")
    ax.set_ylabel("Количество")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_image_rows(
    rows: Sequence[np.ndarray],
    row_labels: Sequence[str],
    *,
    save_path: Path,
    title: str,
    dpi: int = 160,
    show: bool = False,
    cmap: str = "gray",
    max_cols: int = 10,
) -> None:
    rows = [np.asarray(r) for r in rows]
    if len(rows) != len(row_labels):
        raise ValueError("rows and row_labels must have the same length")

    norm_rows = []
    for row in rows:
        if row.ndim == 2 and row.shape[1] == 784:
            row = row.reshape(-1, 28, 28)
        elif row.ndim == 4 and row.shape[1] == 1:
            row = row[:, 0]
        elif row.ndim != 3:
            raise ValueError(f"Unsupported row shape: {row.shape}")
        norm_rows.append(row)

    n_rows = len(norm_rows)
    n_cols = min(max_cols, min(row.shape[0] for row in norm_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.45 * n_cols, 1.65 * n_rows),
        squeeze=False,
    )

    for r, (row, label) in enumerate(zip(norm_rows, row_labels)):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(row[c], cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
        axes[r, 0].set_ylabel(label, fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_image_strip(
    images: np.ndarray,
    *,
    save_path: Path,
    title: str,
    subtitles: Sequence[str] | None = None,
    dpi: int = 160,
    show: bool = False,
    cmap: str = "gray",
) -> None:
    images = np.asarray(images)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    if images.ndim != 3:
        raise ValueError(f"Unsupported strip shape: {images.shape}")

    n = images.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.45 * n, 2.1), squeeze=False)

    for i in range(n):
        ax = axes[0, i]
        ax.imshow(images[i], cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        if subtitles is not None:
            ax.set_title(subtitles[i], fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_line_curve(
    y: Sequence[float],
    *,
    save_path: Path,
    title: str,
    ylabel: str,
    dpi: int = 160,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(np.arange(1, len(y) + 1), y, marker="o", markersize=4)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_dual_curve(
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    label1: str,
    label2: str,
    save_path: Path,
    title: str,
    ylabel: str = "Loss",
    dpi: int = 160,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.plot(np.arange(1, len(y1) + 1), y1, marker="o", markersize=4, label=label1)
    ax.plot(np.arange(1, len(y2) + 1), y2, marker="o", markersize=4, label=label2)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    save_path: Path,
    title: str,
    ylabel: str = "MSE",
    dpi: int = 160,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bars = ax.bar(labels, values, width=0.62)
    ymax = max(values) if len(values) else 1.0

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_scatter_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int = 160,
    show: bool = False,
    cmap: str = "tab10",
    s: float = 6,
    alpha: float = 0.75,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, s=s, alpha=alpha, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(10))
    cbar.set_label("Цифра")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)


def plot_explained_variance(
    cumulative_variance: np.ndarray,
    *,
    save_path: Path,
    title: str,
    dpi: int = 160,
    show: bool = False,
) -> None:
    xs = np.arange(1, len(cumulative_variance) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(xs, cumulative_variance, marker="o", markersize=4)
    ax.axhline(0.9, linestyle="--", linewidth=1, alpha=0.8, label="90%")
    ax.axhline(0.95, linestyle="--", linewidth=1, alpha=0.8, label="95%")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Накопленная доля дисперсии")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, show=show)