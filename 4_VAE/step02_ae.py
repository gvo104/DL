from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from pipeline import cpu_state_dict, load_pickle, save_pickle, plot_image_rows, plot_line_curve


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def _flatten(imgs: torch.Tensor) -> torch.Tensor:
    return imgs.view(imgs.size(0), -1)


def train_ae(
    model: Autoencoder,
    loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_n = 0

        for imgs, _ in loader:
            x = _flatten(imgs).to(device)
            recon, _ = model(x)
            loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        avg_loss = total_loss / total_n
        history.append(avg_loss)
        print(f"  AE epoch {epoch + 1:02d}/{epochs} | loss={avg_loss:.4f}")

    return history


def _valid_cache(cache: object, cfg: Config) -> bool:
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("cache_version") == cfg.cache_version
        and cache.get("model_type") == "AE"
        and cache.get("model_kwargs", {}).get("latent_dim") == cfg.ae_latent_dim
    )


def _load_or_train(cfg: Config, device: torch.device, train_loader):
    cache_path = cfg.cache_dir / "02_ae.pkl"

    if cache_path.exists() and not cfg.force_retrain:
        cache = load_pickle(cache_path)
        if _valid_cache(cache, cfg):
            model = Autoencoder(**cache["model_kwargs"]).to(device)
            model.load_state_dict(cache["state_dict"])
            print("AE loaded from cache.")
            return model, list(cache["history"])
        print("AE cache is incompatible. Retraining...")

    model = Autoencoder(latent_dim=cfg.ae_latent_dim).to(device)
    history = train_ae(model, train_loader, device, epochs=cfg.ae_epochs, lr=cfg.lr)

    save_pickle(
        {
            "cache_version": cfg.cache_version,
            "model_type": "AE",
            "model_kwargs": {"input_dim": 784, "latent_dim": cfg.ae_latent_dim},
            "state_dict": cpu_state_dict(model),
            "history": history,
        },
        cache_path,
    )
    print("AE cache saved.")
    return model, history


@torch.no_grad()
def _reconstruct(model: Autoencoder, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs, recons = [], []

    for imgs, _ in loader:
        x = _flatten(imgs).to(device)
        recon, _ = model(x)
        xs.append(x.cpu().numpy())
        recons.append(recon.cpu().numpy())

    return np.concatenate(xs, axis=0), np.concatenate(recons, axis=0)


def run(cfg: Config, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model, history = _load_or_train(cfg, device, data["train_loader"])

    x_true, x_recon = _reconstruct(model, data["test_loader"], device)
    mse = float(np.mean((x_true - x_recon) ** 2))

    originals = x_true[:10].reshape(-1, 28, 28)
    reconstructions = x_recon[:10].reshape(-1, 28, 28)

    plot_image_rows(
        [originals, reconstructions],
        ["Оригинал", "AE"],
        save_path=cfg.figures_dir / "02_ae_reconstruction.png",
        title="Autoencoder: оригинал и реконструкция",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    plot_line_curve(
        history,
        save_path=cfg.figures_dir / "02_ae_loss_curve.png",
        title="Autoencoder: кривая обучения",
        ylabel="BCE Loss",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    print(f"AE test MSE: {mse:.6f}")

    return {
        "model": model,
        "history": history,
        "test_x": x_true,
        "test_recon": x_recon,
        "mse": mse,
    }