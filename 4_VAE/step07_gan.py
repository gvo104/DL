from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from pipeline import cpu_state_dict, load_pickle, save_pickle


# ======================================================================
# Models
# ======================================================================
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Training
# ======================================================================
def _train_gan(
    generator: Generator,
    discriminator: Discriminator,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
) -> tuple[list[float], list[float]]:
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=cfg.gan_lr, betas=cfg.gan_betas
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.gan_lr, betas=cfg.gan_betas
    )
    criterion = nn.BCELoss()

    g_losses: list[float] = []
    d_losses: list[float] = []

    for epoch in range(cfg.gan_epochs):
        generator.train()
        discriminator.train()
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0
        n_batches = 0

        for imgs, _ in loader:
            batch_size = imgs.size(0)
            real = imgs.view(batch_size, -1).to(device)

            # Label smoothing
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------- Discriminator ----------
            z = torch.randn(batch_size, cfg.gan_latent_dim, device=device)
            fake = generator(z).detach()

            loss_d = (
                criterion(discriminator(real), real_labels)
                + criterion(discriminator(fake), fake_labels)
            ) / 2

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---------- Generator ----------
            z = torch.randn(batch_size, cfg.gan_latent_dim, device=device)
            fake = generator(z)
            loss_g = criterion(discriminator(fake), torch.ones(batch_size, 1, device=device))

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_epoch_loss += loss_g.item()
            d_epoch_loss += loss_d.item()
            n_batches += 1

        g_losses.append(g_epoch_loss / n_batches)
        d_losses.append(d_epoch_loss / n_batches)

        if (epoch + 1) % 10 == 0:
            print(
                f"  GAN epoch {epoch + 1:3d}/{cfg.gan_epochs} | "
                f"G loss: {g_losses[-1]:.4f} | D loss: {d_losses[-1]:.4f}"
            )

    return g_losses, d_losses


# ======================================================================
# Caching
# ======================================================================
def _valid_cache(cache: object, cfg: Config) -> bool:
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("cache_version") == cfg.cache_version
        and cache.get("model_type") == "GAN"
        and cache.get("model_kwargs", {}).get("latent_dim") == cfg.gan_latent_dim
    )


def _load_or_train(
    cfg: Config, device: torch.device, train_loader
) -> tuple[Generator, Discriminator, list[float], list[float]]:
    cache_path = cfg.cache_dir / "07_gan.pkl"

    if cache_path.exists() and not cfg.force_retrain:
        cache = load_pickle(cache_path)
        if _valid_cache(cache, cfg):
            gen = Generator(latent_dim=cfg.gan_latent_dim).to(device)
            disc = Discriminator().to(device)
            gen.load_state_dict(cache["gen_state_dict"])
            disc.load_state_dict(cache["disc_state_dict"])
            print("GAN loaded from cache.")
            return (
                gen, disc,
                list(cache["g_history"]),
                list(cache["d_history"]),
            )
        print("GAN cache incompatible. Retraining...")

    gen = Generator(latent_dim=cfg.gan_latent_dim).to(device)
    disc = Discriminator().to(device)
    g_hist, d_hist = _train_gan(gen, disc, train_loader, device, cfg)

    save_pickle(
        {
            "cache_version": cfg.cache_version,
            "model_type": "GAN",
            "model_kwargs": {"latent_dim": cfg.gan_latent_dim},
            "gen_state_dict": cpu_state_dict(gen),
            "disc_state_dict": cpu_state_dict(disc),
            "g_history": g_hist,
            "d_history": d_hist,
        },
        cache_path,
    )
    print("GAN cache saved.")
    return gen, disc, g_hist, d_hist


# ======================================================================
# Step entry point
# ======================================================================
def run(cfg: Config, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    print("Step 07: GAN")

    gen, disc, g_hist, d_hist = _load_or_train(cfg, device, data["train_loader"])

    # ----- Training curves -----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_hist, label="Generator loss", color="tomato")
    ax.plot(d_hist, label="Discriminator loss", color="steelblue")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="D=0.5 (равновесие)")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.set_title("GAN: кривые обучения")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "07_gan_loss.png", dpi=cfg.fig_dpi)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    final_g = g_hist[-1]
    final_d = d_hist[-1]
    print(f"Финальные потери: G={final_g:.4f}, D={final_d:.4f}")
    if abs(final_d - 0.5) < 0.1:
        print("D ≈ 0.5 — GAN сошёлся к равновесию Нэша.")
    else:
        print("D далёк от 0.5 — возможно, нужно больше эпох.")

    # ----- Random generation -----
    gen.eval()
    with torch.no_grad():
        z = torch.randn(10, cfg.gan_latent_dim, device=device)
        gan_imgs = gen(z).cpu().view(10, 28, 28)

    # Compare with VAE (2D) if available – we'll leave it for step08
    # but we can still save the generated images for later
    save_pickle(
        {"gan_imgs": gan_imgs.numpy()},
        cfg.cache_dir / "07_gan_samples.pkl",
    )

    # ----- Interpolation -----
    with torch.no_grad():
        z_a = torch.randn(1, cfg.gan_latent_dim, device=device)
        z_b = torch.randn(1, cfg.gan_latent_dim, device=device)
        alphas = torch.linspace(0, 1, 10, device=device)
        interp_z = torch.stack(
            [a * z_b + (1 - a) * z_a for a in alphas]
        ).squeeze(1)
        interp_gan = gen(interp_z).cpu().view(10, 28, 28)

    fig, axes = plt.subplots(1, 10, figsize=(14, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interp_gan[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=8)
    fig.suptitle("GAN: интерполяция в латентном пространстве", fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "07_gan_interpolation.png", dpi=cfg.fig_dpi)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    return {
        "generator": gen,
        "discriminator": disc,
        "g_history": g_hist,
        "d_history": d_hist,
        "gan_imgs": gan_imgs,
        "interp_gan": interp_gan,
    }