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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST AE/VAE pipeline")
    parser.add_argument("--show-plots", action="store_true", help="Показывать графики на экране")
    parser.add_argument("--force-retrain", action="store_true", help="Игнорировать кэш и переобучить модели")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = get_config(
        show_plots=args.show_plots,
        force_retrain=args.force_retrain,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    device = prepare_environment(cfg.output_dir, cfg.cache_dir, cfg.figures_dir, cfg.data_dir)

    print(f"Device: {device}")
    print("Step 01: loading data")
    data = run_step01(cfg)

    print("\nStep 02: training / loading AE")
    ae_result = run_step02(cfg, data, device)

    print("\nStep 03: training / loading VAE")
    vae_result = run_step03(cfg, data, device)

    print("\nStep 04: generation experiments")
    _ = run_step04(cfg, data, vae_result["model"], device)

    print("\nStep 05: comparison")
    _ = run_step05(cfg, data, ae_result["model"], vae_result["model"], device)
    
    print("\nStep 06: improve")
    _ = run_step06(cfg, data, device)

    print("\nDone. Figures and caches are saved in output/.")


if __name__ == "__main__":
    main()