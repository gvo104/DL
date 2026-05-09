from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---------- Paths ----------
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)

    # ---------- Reproducibility ----------
    seed: int = 42

    # ---------- Cache ----------
    cache_version: int = 4            # bumped for vae dim list

    # ---------- Data ----------
    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = False
    small_subset_size: int = 5000

    # ---------- Augmentation ----------
    rotation_deg: int = 15
    translate_frac: float = 0.1

    # ---------- Standard AE ----------
    ae_latent_dim: int = 64
    ae_epochs: int = 20

    # ---------- Standard VAE (multiple latent dims) ----------
    vae_latent_dims: tuple[int, ...] = (2, 8, 16)
    vae_epochs: int = 20

    # PCA – auto‑fit to reach variance threshold
    pca_auto_components: bool = True
    pca_variance_threshold: float = 0.95
    pca_fallback_components: int = 64
    pca_ae_latent_dim: int = 16
    pca_ae_epochs: int = 10

    lr: float = 1e-3

    # ---------- Improved β‑VAE (Step 06) ----------
    improved_latent_dim: int = 8
    improved_vae_epochs: int = 20
    beta_max: float = 1.0
    anneal_epochs: int = 10
    
    # ---------- GAN (Step 07) ----------
    gan_latent_dim: int = 64
    gan_epochs: int = 100
    gan_lr: float = 2e-4
    gan_betas: tuple[float, float] = (0.5, 0.999)

    # ---------- Generation & visualisation ----------
    noise_factor: float = 0.3
    latent_grid_min: float = -3.0
    latent_grid_max: float = 3.0
    latent_grid_size: int = 10
    interpolation_steps: int = 10
    noise_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    generated_samples: int = 10

    fig_dpi: int = 160
    show_plots: bool = False

    # ---------- Runtime ----------
    force_retrain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", self.project_root / "data")
        object.__setattr__(self, "output_dir", self.project_root / "output")
        object.__setattr__(self, "cache_dir", self.output_dir / "cache")
        object.__setattr__(self, "figures_dir", self.output_dir / "figures")


def get_config(
    *,
    show_plots: bool = False,
    force_retrain: bool = False,
    seed: int = 42,
) -> Config:
    return Config(
        seed=seed,
        show_plots=show_plots,
        force_retrain=force_retrain,
    )