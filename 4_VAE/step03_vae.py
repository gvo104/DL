from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from pipeline import cpu_state_dict, load_pickle, save_pickle, plot_dual_curve, plot_image_rows


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_h = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_h(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def _flatten(imgs: torch.Tensor) -> torch.Tensor:
    return imgs.view(imgs.size(0), -1)


def vae_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = (bce + kl) / x.size(0)
    return loss, bce / x.size(0), kl / x.size(0)


def train_vae(
    model: VAE,
    loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_history: list[float] = []
    kl_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_bce = 0.0
        total_kl = 0.0
        total_n = 0

        for imgs, _ in loader:
            x = _flatten(imgs).to(device)
            recon, mu, log_var = model(x)
            loss, bce, kl = vae_loss_fn(recon, x, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += bce.item() * x.size(0)
            total_kl += kl.item() * x.size(0)
            total_n += x.size(0)

        avg_bce = total_bce / total_n
        avg_kl = total_kl / total_n
        bce_history.append(avg_bce)
        kl_history.append(avg_kl)
        print(f"  VAE epoch {epoch + 1:02d}/{epochs} | BCE={avg_bce:.2f} | KL={avg_kl:.2f}")

    return bce_history, kl_history


def _valid_cache(cache: object, cfg: Config) -> bool:
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("cache_version") == cfg.cache_version
        and cache.get("model_type") == "VAE"
        and cache.get("model_kwargs", {}).get("latent_dim") == cfg.vae_latent_dim
    )


def _load_or_train(cfg: Config, device: torch.device, train_loader):
    cache_path = cfg.cache_dir / "03_vae_dim2.pkl"

    if cache_path.exists() and not cfg.force_retrain:
        cache = load_pickle(cache_path)
        if _valid_cache(cache, cfg):
            model = VAE(**cache["model_kwargs"]).to(device)
            model.load_state_dict(cache["state_dict"])
            print("VAE loaded from cache.")
            return model, list(cache["bce_history"]), list(cache["kl_history"])
        print("VAE cache is incompatible. Retraining...")

    model = VAE(latent_dim=cfg.vae_latent_dim).to(device)
    bce_history, kl_history = train_vae(model, train_loader, device, epochs=cfg.vae_epochs, lr=cfg.lr)

    save_pickle(
        {
            "cache_version": cfg.cache_version,
            "model_type": "VAE",
            "model_kwargs": {"input_dim": 784, "latent_dim": cfg.vae_latent_dim},
            "state_dict": cpu_state_dict(model),
            "bce_history": bce_history,
            "kl_history": kl_history,
        },
        cache_path,
    )
    print("VAE cache saved.")
    return model, bce_history, kl_history


@torch.no_grad()
def _reconstruct(model: VAE, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs, recons = [], []

    for imgs, _ in loader:
        x = _flatten(imgs).to(device)
        recon, _, _ = model(x)
        xs.append(x.cpu().numpy())
        recons.append(recon.cpu().numpy())

    return np.concatenate(xs, axis=0), np.concatenate(recons, axis=0)


@torch.no_grad()
def encode_mu(model: VAE, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus, labels = [], []

    for imgs, y in loader:
        x = _flatten(imgs).to(device)
        mu, _ = model.encode(x)
        mus.append(mu.cpu().numpy())
        labels.append(y.numpy())

    return np.concatenate(mus, axis=0), np.concatenate(labels, axis=0)


def run(cfg: Config, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model, bce_history, kl_history = _load_or_train(cfg, device, data["train_loader"])

    x_true, x_recon = _reconstruct(model, data["test_loader"], device)
    mse = float(np.mean((x_true - x_recon) ** 2))

    originals = x_true[:10].reshape(-1, 28, 28)
    reconstructions = x_recon[:10].reshape(-1, 28, 28)

    plot_image_rows(
        [originals, reconstructions],
        ["Оригинал", "VAE"],
        save_path=cfg.figures_dir / "03_vae_reconstruction_dim2.png",
        title="VAE: оригинал и реконструкция",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    plot_dual_curve(
        bce_history,
        kl_history,
        label1="BCE",
        label2="KL",
        save_path=cfg.figures_dir / "03_vae_loss_curve_dim2.png",
        title="VAE: кривые обучения",
        ylabel="Loss",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    print(f"VAE test MSE: {mse:.6f}")

    return {
        "model": model,
        "bce_history": bce_history,
        "kl_history": kl_history,
        "test_x": x_true,
        "test_recon": x_recon,
        "mse": mse,
    }