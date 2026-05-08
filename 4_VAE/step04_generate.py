from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from pipeline import plot_image_rows, save_pickle


@torch.no_grad()
def run(cfg: Config, data: dict[str, Any], vae_model,
        device: torch.device) -> dict[str, Any]:
    vae_model.eval()

    x_test = data["x_test"]

    # ----- Random generation -----
    z = torch.randn(cfg.generated_samples, vae_model.latent_dim, device=device)
    random_imgs = vae_model.decode(z).cpu().numpy().reshape(-1, 28, 28)

    plot_image_rows(
        [x_test[:cfg.generated_samples], random_imgs],
        ["Оригинал (test)", "Случайные сэмплы из N(0,I)"],
        save_path=cfg.figures_dir / "04_random_generation.png",
        title="VAE (2D): случайная генерация",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # ----- Interpolation -----
    point_a = torch.randn(1, vae_model.latent_dim, device=device)
    point_b = torch.randn(1, vae_model.latent_dim, device=device)
    alphas = torch.linspace(0, 1, cfg.interpolation_steps, device=device)

    interp_z = torch.stack(
        [(1 - a) * point_a + a * point_b for a in alphas], dim=0
    ).squeeze(1)
    interp_imgs = vae_model.decode(interp_z).cpu().numpy().reshape(
        cfg.interpolation_steps, 28, 28
    )

    fig, axes = plt.subplots(
        1, cfg.interpolation_steps,
        figsize=(1.45 * cfg.interpolation_steps, 2.15),
        squeeze=False,
    )
    for i, ax in enumerate(axes[0]):
        ax.imshow(interp_imgs[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"α={float(alphas[i]):.1f}", fontsize=9)
    fig.suptitle(
        "VAE (2D): интерполяция между двумя точками латентного пространства",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        cfg.figures_dir / "04_interpolation.png",
        dpi=cfg.fig_dpi,
        bbox_inches="tight",
    )
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    # ----- 2D latent grid -----
    grid_x = np.linspace(cfg.latent_grid_min, cfg.latent_grid_max, cfg.latent_grid_size)
    grid_y = np.linspace(cfg.latent_grid_min, cfg.latent_grid_max, cfg.latent_grid_size)
    latent_grid = np.array(
        [[x, y] for y in grid_y for x in grid_x], dtype=np.float32
    )

    grid_imgs = (
        vae_model.decode(torch.tensor(latent_grid, device=device))
        .cpu()
        .numpy()
        .reshape(cfg.latent_grid_size, cfg.latent_grid_size, 28, 28)
    )

    fig, axes = plt.subplots(
        cfg.latent_grid_size, cfg.latent_grid_size,
        figsize=(10, 10),
        squeeze=False,
    )
    for r in range(cfg.latent_grid_size):
        for c in range(cfg.latent_grid_size):
            axes[r, c].imshow(grid_imgs[r, c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
    fig.suptitle(
        "VAE (2D): декодированные изображения из сетки латентного пространства\n"
        f"(диапазон z₁, z₂ ∈ [{cfg.latent_grid_min}, {cfg.latent_grid_max}])",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        cfg.figures_dir / "04_latent_grid.png",
        dpi=cfg.fig_dpi,
        bbox_inches="tight",
    )
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    # ----- Noise experiment -----
    base_z = torch.randn(8, vae_model.latent_dim, device=device)
    noise_rows = []
    for sigma in cfg.noise_levels:
        noisy_z = base_z + sigma * torch.randn_like(base_z)
        imgs = vae_model.decode(noisy_z).cpu().numpy().reshape(-1, 28, 28)
        noise_rows.append(imgs)

    plot_image_rows(
        noise_rows,
        [f"шум σ = {sigma:.1f}" for sigma in cfg.noise_levels],
        save_path=cfg.figures_dir / "04_noise_experiment.png",
        title="VAE (2D): влияние аддитивного шума в латентном пространстве",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    cache = {
        "random_imgs": random_imgs,
        "interp_imgs": interp_imgs,
        "grid_imgs": grid_imgs,
        "noise_levels": cfg.noise_levels,
        "noise_rows": noise_rows,
    }
    save_pickle(cache, cfg.cache_dir / "04_generate_dim2.pkl")
    return cache