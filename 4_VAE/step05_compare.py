from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from pipeline import (
    plot_bar_comparison,
    plot_explained_variance,
    plot_image_rows,
    save_pickle,
)


class AEonPCA(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def _train_pca_ae(cfg: Config, x_train_pca: np.ndarray,
                  device: torch.device) -> AEonPCA:
    dataset = TensorDataset(
        torch.tensor(x_train_pca, dtype=torch.float32),
        torch.zeros(len(x_train_pca), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = AEonPCA(
        input_dim=cfg.pca_components_used,  # will be set before calling
        latent_dim=cfg.pca_ae_latent_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.pca_ae_epochs):
        model.train()
        total = 0.0
        n_batches = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            recon, _ = model(x_batch)
            loss = F.mse_loss(recon, x_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n_batches += 1

        print(
            f"  PCA+AE epoch {epoch + 1:02d}/{cfg.pca_ae_epochs} "
            f"| mse={total / n_batches:.6f}"
        )

    return model


def _determine_pca_components(
    x_train: np.ndarray, cfg: Config
) -> tuple[int, np.ndarray, PCA]:
    """
    Fit a full PCA (up to min(n_samples, n_features)) and return:
    - number of components needed to reach cfg.pca_variance_threshold
    - cumulative explained variance array
    - the PCA object fitted with that number of components
    """
    max_comp = min(x_train.shape[0], x_train.shape[1])  # 60000 × 784 → 784
    pca_full = PCA(n_components=max_comp, svd_solver="full", random_state=cfg.seed)
    pca_full.fit(x_train)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumsum, cfg.pca_variance_threshold) + 1)
    n_needed = min(n_needed, max_comp)
    print(
        f"PCA: {n_needed} компонент объясняют ≥ {cfg.pca_variance_threshold*100:.0f}% дисперсии "
        f"(отобрано из {max_comp})"
    )
    # Refit with exactly n_needed components for the rest of the pipeline
    pca = PCA(n_components=n_needed, svd_solver="full", random_state=cfg.seed)
    pca.fit(x_train)
    return n_needed, cumsum, pca


def run(cfg: Config, data: dict[str, Any], ae_model, vae_model,
        device: torch.device) -> dict[str, Any]:
    x_train = data["x_train"].reshape(len(data["x_train"]), -1).astype(np.float32)
    x_test = data["x_test"].reshape(len(data["x_test"]), -1).astype(np.float32)

    # ---------- PCA (auto components) ----------
    if cfg.pca_auto_components:
        n_comp, explained, pca64 = _determine_pca_components(x_train, cfg)
        cfg.pca_components_used = n_comp
    else:
        n_comp = cfg.pca_fallback_components
        pca64 = PCA(n_components=n_comp, svd_solver="randomized", random_state=cfg.seed)
        pca64.fit(x_train)
        explained = np.cumsum(pca64.explained_variance_ratio_)
        cfg.pca_components_used = n_comp

    x_train_pca64 = pca64.transform(x_train)
    x_test_pca64 = pca64.transform(x_test)
    x_test_pca64_recon = pca64.inverse_transform(x_test_pca64)
    mse_pca = mean_squared_error(x_test, x_test_pca64_recon)

    plot_explained_variance(
        explained,
        save_path=cfg.figures_dir / "05_pca_explained_variance.png",
        title=(
            f"PCA: накопленная объяснённая дисперсия\n"
            f"(использовано {n_comp} компонент, порог {cfg.pca_variance_threshold*100:.0f}%)"
        ),
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # ---------- AE ----------
    ae_model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        x_test_ae_recon = ae_model(x_t)[0].cpu().numpy()
    mse_ae = mean_squared_error(x_test, x_test_ae_recon)

    # ---------- VAE ----------
    vae_model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        x_test_vae_recon = vae_model(x_t)[0].cpu().numpy()
    mse_vae = mean_squared_error(x_test, x_test_vae_recon)

    # ---------- PCA + AE ----------
    pca_ae_model = _train_pca_ae(cfg, x_train_pca64.astype(np.float32), device)
    pca_ae_model.eval()
    with torch.no_grad():
        x_pca_t = torch.tensor(x_test_pca64.astype(np.float32), device=device)
        x_test_pca_ae_recon_pca = pca_ae_model(x_pca_t)[0].cpu().numpy()
        x_test_pca_ae_recon = pca64.inverse_transform(x_test_pca_ae_recon_pca)
    mse_pca_ae = mean_squared_error(x_test, x_test_pca_ae_recon)

    # ---------- Comparison ----------
    labels = [
        f"PCA ({n_comp}d)",
        f"AE ({cfg.ae_latent_dim}d)",
        f"VAE ({cfg.vae_latent_dim}d)",
        "PCA+AE",
    ]
    mses = [mse_pca, mse_ae, mse_vae, mse_pca_ae]

    plot_bar_comparison(
        labels,
        mses,
        save_path=cfg.figures_dir / "05_compare_mse.png",
        title="Сравнение методов по ошибке реконструкции (MSE)",
        ylabel="MSE",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # Reconstruction grid
    row_labels = [
        "Оригинал",
        f"PCA (MSE={mse_pca:.5f})",
        f"AE (MSE={mse_ae:.5f})",
        f"VAE (MSE={mse_vae:.5f})",
        f"PCA+AE (MSE={mse_pca_ae:.5f})",
    ]
    reconstruction_rows = [
        x_test[:10].reshape(-1, 28, 28),
        x_test_pca64_recon[:10].reshape(-1, 28, 28),
        x_test_ae_recon[:10].reshape(-1, 28, 28),
        x_test_vae_recon[:10].reshape(-1, 28, 28),
        x_test_pca_ae_recon[:10].reshape(-1, 28, 28),
    ]
    plot_image_rows(
        reconstruction_rows,
        row_labels,
        save_path=cfg.figures_dir / "05_reconstruction_compare.png",
        title="Сравнение реконструкций",
        dpi=cfg.fig_dpi,
        show=cfg.show_plots,
    )

    # ---------- Latent space comparison ----------
    pca2 = PCA(n_components=2, svd_solver="randomized", random_state=cfg.seed)
    x_train_pca2 = pca2.fit_transform(x_train)

    train_loader_for_plot = DataLoader(
        data["train_dataset"],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )

    mus = []
    labels_all = []
    with torch.no_grad():
        for imgs, y in train_loader_for_plot:
            x = imgs.view(imgs.size(0), -1).to(device)
            mu, _ = vae_model.encode(x)
            mus.append(mu.cpu().numpy())
            labels_all.append(y.numpy())

    mu_train = np.concatenate(mus, axis=0)
    y_train = np.concatenate(labels_all, axis=0)

    take = min(cfg.small_subset_size, len(x_train_pca2), len(mu_train))

    # Use constrained_layout to avoid colorbar/axes overlap
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.8),
        squeeze=False,
        constrained_layout=True,
    )

    sc1 = axes[0, 0].scatter(
        x_train_pca2[:take, 0], x_train_pca2[:take, 1],
        c=y_train[:take], cmap="tab10", s=6, alpha=0.75, linewidths=0,
    )
    axes[0, 0].set_title("PCA (2 компоненты)")
    axes[0, 0].set_xlabel("PC 1")
    axes[0, 0].set_ylabel("PC 2")
    axes[0, 0].grid(True)

    sc2 = axes[0, 1].scatter(
        mu_train[:take, 0], mu_train[:take, 1],
        c=y_train[:take], cmap="tab10", s=6, alpha=0.75, linewidths=0,
    )
    axes[0, 1].set_title("VAE (μ пространство, 2D)")
    axes[0, 1].set_xlabel("z₁")
    axes[0, 1].set_ylabel("z₂")
    axes[0, 1].grid(True)

    cbar = fig.colorbar(
        sc2, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02,
    )
    cbar.set_label("Цифра")
    fig.suptitle("Структура латентного пространства: PCA vs VAE", fontsize=13)

    fig.savefig(
        cfg.figures_dir / "05_latent_space_compare.png",
        dpi=cfg.fig_dpi,
        bbox_inches="tight",
    )
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    # ---------- Summary ----------
    results = pd.DataFrame({
        "Метод": labels,
        "Latent dim": [n_comp, cfg.ae_latent_dim, cfg.vae_latent_dim, cfg.pca_ae_latent_dim],
        "MSE": [mse_pca, mse_ae, mse_vae, mse_pca_ae],
    })

    save_pickle(
        {
            "results": results,
            "mse_pca": mse_pca,
            "mse_ae": mse_ae,
            "mse_vae": mse_vae,
            "mse_pca_ae": mse_pca_ae,
            "explained_variance": explained,
            "pca_components_used": n_comp,
        },
        cfg.cache_dir / "05_compare.pkl",
    )

    print("\nИтоговая таблица:")
    print(results.to_string(index=False))
    print(
        f"PCA: {n_comp} компонент объясняют "
        f"{explained[n_comp-1] * 100:.2f}% дисперсии"
    )
    print(f"MSE PCA    : {mse_pca:.6f}")
    print(f"MSE AE     : {mse_ae:.6f}")
    print(f"MSE VAE    : {mse_vae:.6f}")
    print(f"MSE PCA+AE : {mse_pca_ae:.6f}")

    # Attach pca_components_used to cfg for step06 (final comparison)
    cfg.pca_components_used = n_comp
    return {
        "results": results,
        "pca": pca64,
        "pca2": pca2,
        "mse_pca": mse_pca,
        "mse_ae": mse_ae,
        "mse_vae": mse_vae,
        "mse_pca_ae": mse_pca_ae,
        "x_test_pca64_recon": x_test_pca64_recon,
        "x_test_ae_recon": x_test_ae_recon,
        "x_test_vae_recon": x_test_vae_recon,
        "x_test_pca_ae_recon": x_test_pca_ae_recon,
    }