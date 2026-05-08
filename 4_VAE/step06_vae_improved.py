from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from config import Config
from pipeline import (
    cpu_state_dict,
    load_pickle,
    save_pickle,
    plot_bar_comparison,
    plot_image_rows,
    plot_training_curves_with_beta,
)


class BetaVAE(nn.Module):
    """Improved VAE with deeper architecture and KL annealing."""
    def __init__(self, input_dim: int = 784, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _loss_fn(
    recon: torch.Tensor, x: torch.Tensor,
    mu: torch.Tensor, logvar: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (bce + beta * kl) / x.size(0)
    return loss, bce / x.size(0), kl / x.size(0)


def _flatten(imgs: torch.Tensor) -> torch.Tensor:
    return imgs.view(imgs.size(0), -1)


def _train(
    model: BetaVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: dict[str, list[float]] = {"bce": [], "kl": [], "beta": []}

    for epoch in range(cfg.improved_vae_epochs):
        model.train()
        beta = min(cfg.beta_max, (epoch + 1) / cfg.anneal_epochs * cfg.beta_max)

        total_bce = 0.0
        total_kl = 0.0
        total_n = 0

        for imgs, _ in loader:
            x = _flatten(imgs).to(device)
            recon, mu, logvar = model(x)
            loss, bce, kl = _loss_fn(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += bce.item() * x.size(0)
            total_kl += kl.item() * x.size(0)
            total_n += x.size(0)

        avg_bce = total_bce / total_n
        avg_kl = total_kl / total_n
        history["bce"].append(avg_bce)
        history["kl"].append(avg_kl)
        history["beta"].append(beta)

        print(
            f"  β-VAE epoch {epoch + 1:02d}/{cfg.improved_vae_epochs} | "
            f"BCE={avg_bce:.2f} | KL={avg_kl:.2f} | β={beta:.3f}"
        )

    return history


def _valid_cache(cache: object, cfg: Config) -> bool:
    if not isinstance(cache, dict):
        return False
    return (
        cache.get("cache_version") == cfg.cache_version
        and cache.get("model_type") == "BetaVAE"
        and cache.get("model_kwargs", {}).get("latent_dim") == cfg.improved_latent_dim
        and cache.get("beta_max") == cfg.beta_max
        and cache.get("anneal_epochs") == cfg.anneal_epochs
    )


def _load_or_train(
    cfg: Config, device: torch.device, train_loader
) -> tuple[BetaVAE, dict[str, list[float]]]:
    cache_path = cfg.cache_dir / "06_improved_vae.pkl"

    if cache_path.exists() and not cfg.force_retrain:
        cache = load_pickle(cache_path)
        if _valid_cache(cache, cfg):
            model = BetaVAE(**cache["model_kwargs"]).to(device)
            model.load_state_dict(cache["state_dict"])
            print("Improved β-VAE loaded from cache.")
            return model, {
                "bce": cache["bce_history"],
                "kl": cache["kl_history"],
                "beta": cache["beta_history"],
            }
        print("β-VAE cache is incompatible. Retraining...")

    model = BetaVAE(latent_dim=cfg.improved_latent_dim).to(device)
    history = _train(model, train_loader, device, cfg)

    save_pickle(
        {
            "cache_version": cfg.cache_version,
            "model_type": "BetaVAE",
            "model_kwargs": {"input_dim": 784, "latent_dim": cfg.improved_latent_dim},
            "state_dict": cpu_state_dict(model),
            "bce_history": history["bce"],
            "kl_history": history["kl"],
            "beta_history": history["beta"],
            "beta_max": cfg.beta_max,
            "anneal_epochs": cfg.anneal_epochs,
        },
        cache_path,
    )
    print("β-VAE cache saved.")
    return model, history


@torch.no_grad()
def _compute_mse(model: BetaVAE, loader, device: torch.device) -> float:
    model.eval()
    total_mse = 0.0
    total_n = 0
    for imgs, _ in loader:
        x = _flatten(imgs).to(device)
        recon, _, _ = model(x)
        mse = F.mse_loss(recon, x, reduction="sum")
        total_mse += mse.item()
        total_n += x.size(0)
    return total_mse / total_n


@torch.no_grad()
def _encode_mu(model: BetaVAE, loader, device: torch.device):
    model.eval()
    mus, labels = [], []
    for imgs, y in loader:
        x = _flatten(imgs).to(device)
        mu, _ = model.encode(x)
        mus.append(mu.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(mus, axis=0), np.concatenate(labels, axis=0)


def _plot_latent_space(
    mu: np.ndarray,
    labels: np.ndarray,
    save_path,
    title: str,
    dpi: int,
    show: bool,
    max_points: int = 5000,
) -> None:
    take = min(max_points, len(mu))
    # Reduce to 2D with PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    mu_2d = pca.fit_transform(mu[:take])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        mu_2d[:, 0], mu_2d[:, 1],
        c=labels[:take], cmap="tab10", s=6, alpha=0.75, linewidths=0,
    )
    ax.set_xlabel("PC 1 (проекция μ)")
    ax.set_ylabel("PC 2 (проекция μ)")
    ax.set_title(title)
    ax.grid(True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Цифра")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run(cfg: Config, data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    print("Step 06: Improved β-VAE with annealing")

    model, history = _load_or_train(cfg, device, data["train_loader"])

    # Training curves with beta
    plot_training_curves_with_beta(
        history["bce"],
        history["kl"],
        history["beta"],
        save_path=cfg.figures_dir / "06_training.png",
        title="β-VAE training dynamics",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # Random generation preview
    x_test_sample = data["x_test"][:10]
    z = torch.randn(25, cfg.improved_latent_dim, device=device)
    model.eval()
    with torch.no_grad():
        gen_imgs = model.decode(z).cpu().view(-1, 28, 28).numpy()
    plot_image_rows(
        [x_test_sample, gen_imgs],
        ["Оригинал (test)", "β-VAE генерация"],
        save_path=cfg.figures_dir / "06_generation.png",
        title="β-VAE: случайная генерация",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
        max_cols=10,
    )

    # ----- MSE on test set -----
    mse_bvae = _compute_mse(model, data["test_loader"], device)
    print(f"β-VAE test MSE: {mse_bvae:.6f}")

    # ----- Latent space (PCA projection) -----
    print("Computing latent representation for β-VAE...")
    mu_bvae, labels_bvae = _encode_mu(model, data["test_loader"], device)
    _plot_latent_space(
        mu_bvae,
        labels_bvae,
        save_path=cfg.figures_dir / "06_latent_space.png",
        title=f"β‑VAE: латентное пространство (μ, {cfg.improved_latent_dim}D) → PCA(2D)",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # ----- Final comparison with previous methods -----
    # Load previous MSEs from cache if available, otherwise use placeholders
    cache_05_path = cfg.cache_dir / "05_compare.pkl"
    if cache_05_path.exists():
        cache_05 = load_pickle(cache_05_path)
        mse_pca = cache_05.get("mse_pca", float("nan"))
        mse_ae = cache_05.get("mse_ae", float("nan"))
        mse_vae = cache_05.get("mse_vae", float("nan"))
        mse_pca_ae = cache_05.get("mse_pca_ae", float("nan"))
        pca_comp = cache_05.get("pca_components_used", cfg.pca_fallback_components)
    else:
        mse_pca = mse_ae = mse_vae = mse_pca_ae = float("nan")
        pca_comp = cfg.pca_fallback_components

    all_labels = [
        f"PCA ({pca_comp}d)",
        f"AE ({cfg.ae_latent_dim}d)",
        f"VAE ({cfg.vae_latent_dim}d)",
        "PCA+AE",
        f"β-VAE ({cfg.improved_latent_dim}d)",
    ]
    all_mses = [mse_pca, mse_ae, mse_vae, mse_pca_ae, mse_bvae]

    plot_bar_comparison(
        all_labels,
        all_mses,
        save_path=cfg.figures_dir / "06_compare_all.png",
        title="Финальное сравнение MSE всех методов",
        ylabel="MSE",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    return {"model": model, "history": history, "mse": mse_bvae}