from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config
from pipeline import (
    load_numpy,
    save_numpy,
    plot_class_distribution,
    plot_image_rows,
)


def _dataset_to_numpy(dataset: datasets.MNIST) -> tuple[np.ndarray, np.ndarray]:
    x = dataset.data.float().numpy() / 255.0
    y = dataset.targets.numpy()
    return x.astype(np.float32), y.astype(np.int64)


def _load_or_build_arrays(cfg: Config, train_dataset, test_dataset) -> dict[str, np.ndarray]:
    paths = {
        "x_train": cfg.cache_dir / "x_train.npy",
        "x_test": cfg.cache_dir / "x_test.npy",
        "y_train": cfg.cache_dir / "y_train.npy",
        "y_test": cfg.cache_dir / "y_test.npy",
    }

    if (
        not cfg.force_retrain
        and all(path.exists() for path in paths.values())
    ):
        return {
            "x_train": load_numpy(paths["x_train"]).astype(np.float32),
            "x_test": load_numpy(paths["x_test"]).astype(np.float32),
            "y_train": load_numpy(paths["y_train"]).astype(np.int64),
            "y_test": load_numpy(paths["y_test"]).astype(np.int64),
        }

    x_train, y_train = _dataset_to_numpy(train_dataset)
    x_test, y_test = _dataset_to_numpy(test_dataset)

    save_numpy(paths["x_train"], x_train)
    save_numpy(paths["x_test"], x_test)
    save_numpy(paths["y_train"], y_train)
    save_numpy(paths["y_test"], y_test)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run(cfg: Config) -> dict[str, Any]:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    pin_memory = cfg.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    arrays = _load_or_build_arrays(cfg, train_dataset, test_dataset)
    x_train = arrays["x_train"]
    x_test = arrays["x_test"]
    y_train = arrays["y_train"]
    y_test = arrays["y_test"]

    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    summary = {
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "image_shape": tuple(x_train[0].shape),
        "pixel_min": float(x_train.min()),
        "pixel_max": float(x_train.max()),
        "pixel_mean": float(x_train.mean()),
        "pixel_std": float(x_train.std()),
        "train_class_counts": np.bincount(y_train, minlength=10).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=10).astype(int).tolist(),
    }

    from pipeline import save_pickle

    save_pickle(summary, cfg.cache_dir / "01_data_loader.pkl")

    # Preview: original vs augmented
    aug_transform = transforms.Compose(
        [
            transforms.RandomRotation(cfg.rotation_deg),
            transforms.RandomAffine(0, translate=(cfg.translate_frac, cfg.translate_frac)),
            transforms.ToTensor(),
        ]
    )
    aug_dataset = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=False,
        transform=aug_transform,
    )

    n_preview = 10
    originals = np.stack([train_dataset[i][0].squeeze(0).numpy() for i in range(n_preview)], axis=0)
    augmented = np.stack([aug_dataset[i][0].squeeze(0).numpy() for i in range(n_preview)], axis=0)

    plot_image_rows(
        [originals, augmented],
        ["Оригинал", "Аугментация"],
        save_path=cfg.figures_dir / "01_data_preview.png",
        title="MNIST: оригиналы и аугментированные примеры",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    plot_class_distribution(
        summary["train_class_counts"],
        summary["test_class_counts"],
        save_path=cfg.figures_dir / "01_class_distribution.png",
        title="MNIST: распределение классов train / test",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    print(f"Train: {summary['train_size']}  Test: {summary['test_size']}")
    print(
        f"Pixels: min={summary['pixel_min']:.4f}, "
        f"max={summary['pixel_max']:.4f}, "
        f"mean={summary['pixel_mean']:.4f}, "
        f"std={summary['pixel_std']:.4f}"
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_flat": x_train_flat,
        "x_test_flat": x_test_flat,
        "summary": summary,
    }