from __future__ import annotations

import argparse

from config import get_config
from pipeline import prepare_environment, set_seed
from step01_data_loader import run as run_step01
from step02_ae import run as run_step02
from step03_vae import run as run_step03
from step04_generate import run as run_step04
from step05_compare import run as run_step05
from step06_vae_improved import run as run_step06
from step07_gan import run as run_step07
from step08_final_comparison import run as run_step08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST AE/VAE/β-VAE/GAN pipeline")
    parser.add_argument("--show-plots", action="store_true",
                        help="Показывать графики на экране")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Игнорировать кеш и переобучить модели")
    parser.add_argument("--seed", type=int, default=42, help="Глобальный seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = get_config(
        show_plots=args.show_plots,
        force_retrain=args.force_retrain,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = prepare_environment(cfg.output_dir, cfg.cache_dir,
                                 cfg.figures_dir, cfg.data_dir)

    print(f"Device: {device}")

    print("\nStep 01: Loading data")
    data = run_step01(cfg)

    print("\nStep 02: AE")
    ae_result = run_step02(cfg, data, device)

    print("\nStep 03: VAE (multi-dimensional)")
    vae_results = run_step03(cfg, data, device)

    if 2 in vae_results["models"]:
        print("\nStep 04: Generation experiments (VAE 2D)")
        run_step04(cfg, data, vae_results["models"][2], device)
    else:
        print("\nStep 04: SKIPPED (no 2D VAE available)")

    print("\nStep 05: Comparison")
    compare_results = run_step05(cfg, data, ae_result["model"], vae_results, device)

    print("\nStep 06: Improved β-VAE")
    bvae_results = run_step06(cfg, data, device)

    print("\nStep 07: GAN")
    gan_results = run_step07(cfg, data, device)

    print("\nStep 08: Final analysis")
    run_step08(
        cfg, data,
        ae_result, vae_results,
        compare_results, bvae_results,
        gan_results, device,
    )

    print("\nDone. Figures and caches are saved in output/.")


if __name__ == "__main__":
    main()