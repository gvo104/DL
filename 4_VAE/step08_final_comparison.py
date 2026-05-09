from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import Config
from pipeline import save_pickle, load_pickle


def run(cfg: Config, data: dict[str, Any],
        ae_results: dict, vae_results: dict,
        compare_results: dict, bvae_results: dict,
        gan_results: dict, device: Any) -> None:
    print("Step 08: Final comparison")

    # ------------------------------------------------------------------
    # Collect MSEs
    # ------------------------------------------------------------------
    mse_pca = compare_results["mse_pca"]
    mse_ae = compare_results["mse_ae"]
    mse_pca_ae = compare_results["mse_pca_ae"]
    vae_mses = vae_results["mses"]          # dict {2: ..., 8: ..., 16: ...}
    mse_bvae = bvae_results.get("mse", None)

    # GAN does not have reconstruction MSE → we can mark as '-' or skip
    methods = []
    mses = []
    latent_dims = []
    has_generation = []
    interpretability = []

    # PCA
    n_comp = compare_results.get("pca_components_used",
                                 cfg.pca_fallback_components)
    methods.append(f"PCA ({n_comp}d)")
    mses.append(mse_pca)
    latent_dims.append(n_comp)
    has_generation.append("Нет")
    interpretability.append("Высокая")

    # AE
    methods.append(f"AE ({cfg.ae_latent_dim}d)")
    mses.append(mse_ae)
    latent_dims.append(cfg.ae_latent_dim)
    has_generation.append("Нет")
    interpretability.append("Средняя")

    # VAE variants
    for dim in sorted(vae_mses.keys()):
        methods.append(f"VAE ({dim}d)")
        mses.append(vae_mses[dim])
        latent_dims.append(dim)
        has_generation.append("Да")
        interpretability.append(
            "Высокая" if dim == 2 else "Низкая"
        )

    # PCA+AE
    methods.append(f"PCA+AE (lat {cfg.pca_ae_latent_dim}d)")
    mses.append(mse_pca_ae)
    latent_dims.append(cfg.pca_ae_latent_dim)
    has_generation.append("Нет")
    interpretability.append("Средняя")

    # β-VAE
    if mse_bvae is not None:
        methods.append(f"β‑VAE ({cfg.improved_latent_dim}d)")
        mses.append(mse_bvae)
        latent_dims.append(cfg.improved_latent_dim)
        has_generation.append("Да")
        interpretability.append("Средняя (развязанность)")

    # GAN
    # No MSE, we can add a row with a dash
    methods.append(f"GAN ({cfg.gan_latent_dim}d)")
    mses.append(float("nan"))
    latent_dims.append(cfg.gan_latent_dim)
    has_generation.append("Да")
    interpretability.append("Низкая (неявная)")

    # ------------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "Метод": methods,
        "Latent dim": latent_dims,
        "MSE": [f"{v:.5f}" if not np.isnan(v) else "—" for v in mses],
        "Генерация": has_generation,
        "Интерпретируемость": interpretability,
    })

    print("\n=== Итоговая сводка ===")
    print(df.to_string(index=False))
    save_pickle({"final_df": df}, cfg.cache_dir / "08_final_summary.pkl")

    # ------------------------------------------------------------------
    # Plot MSE comparison (skip GAN, which has no MSE)
    # ------------------------------------------------------------------
    plot_methods = [m for m, v in zip(methods, mses) if not np.isnan(v)]
    plot_mses = [v for v in mses if not np.isnan(v)]
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_methods)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(plot_methods, plot_mses, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, plot_mses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(plot_mses) * 0.01,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("MSE (ошибка реконструкции)")
    ax.set_title("Итоговое сравнение методов по качеству реконструкции")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "08_final_mse_comparison.png", dpi=cfg.fig_dpi)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # Side-by-side generation: VAE (2D) vs GAN (if both exist)
    # ------------------------------------------------------------------
    if 2 in vae_results["models"]:
        vae_model = vae_results["models"][2]
        vae_model.eval()
        with torch.no_grad():
            vae_imgs = (
                vae_model.decode(
                    torch.randn(10, 2, device=device)
                ).cpu().view(10, 28, 28)
            )

        gan_gen = gan_results.get("gan_imgs")
        if gan_gen is not None:
            fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
            for i in range(10):
                axes[0, i].imshow(vae_imgs[i], cmap="gray", vmin=0, vmax=1)
                axes[0, i].axis("off")
                axes[1, i].imshow(gan_gen[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                axes[1, i].axis("off")
            axes[0, 0].set_ylabel("VAE (2d)", fontsize=9)
            axes[1, 0].set_ylabel("GAN", fontsize=9)
            fig.suptitle("Сравнение генерации: VAE vs GAN", fontsize=12)
            fig.tight_layout()
            fig.savefig(
                cfg.figures_dir / "08_vae_vs_gan.png",
                dpi=cfg.fig_dpi,
            )
            if cfg.show_plots:
                plt.show()
            plt.close(fig)

    print("Итоговый анализ завершён.")