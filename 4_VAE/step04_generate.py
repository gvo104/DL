import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pipeline import PipelineStep
from step03_vae import sampling   # импорт зарегистрированной функции


class GenerateStep(PipelineStep):
    def __init__(self, config, force=False):
        super().__init__("04_generate", config, force)
        self.decoder_path = os.path.join(config.CACHE_DIR, "vae_decoder.keras")
        self.encoder_path = os.path.join(config.CACHE_DIR, "vae_encoder.keras")
        self.latent_dim = 2

    def run(self):
        # Загружаем модели с указанием custom_objects
        decoder = tf.keras.models.load_model(
            self.decoder_path, compile=False,
            custom_objects={'sampling': sampling}
        )
        encoder = tf.keras.models.load_model(
            self.encoder_path, compile=False,
            custom_objects={'sampling': sampling}
        )

        x_test = np.load(os.path.join(self.config.CACHE_DIR, "x_test.npy"))

        # 1. Случайная генерация и сравнение с оригиналами
        self._random_generation_and_comparison(decoder, x_test)

        # 2. Интерполяция
        self._interpolation(decoder)

        # 3. Сетка латентного пространства
        self._latent_grid(decoder)

        # 4. Добавление шума
        self._noise_experiment(decoder, x_test, encoder)

        return {"status": "completed"}

    def _random_generation_and_comparison(self, decoder, x_test):
        latent_samples = np.random.normal(size=(10, self.latent_dim))
        generated = decoder.predict(latent_samples, verbose=0)

        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].imshow(x_test[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(generated[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_title("Originals (test)")
        axes[1, 0].set_title("Random Generation")
        fig_path = os.path.join(self.config.FIGURES_DIR, "04_random_generation.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Случайная генерация сохранена: {fig_path}")

    def _interpolation(self, decoder):
        point_a = np.random.normal(size=(1, self.latent_dim))
        point_b = np.random.normal(size=(1, self.latent_dim))
        steps = 10
        alphas = np.linspace(0, 1, steps)
        interpolated = point_a * (1 - alphas[:, None]) + point_b * alphas[:, None]
        images = decoder.predict(interpolated, verbose=0)

        fig, axes = plt.subplots(1, steps, figsize=(12, 1.5))
        for i in range(steps):
            axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
        fig_path = os.path.join(self.config.FIGURES_DIR, "04_interpolation.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Интерполяция сохранена: {fig_path}")

    def _latent_grid(self, decoder):
        grid_size = 10
        axis_range = np.linspace(-3, 3, grid_size)
        grid = np.array([[x, y] for x in axis_range for y in axis_range])
        images = decoder.predict(grid, verbose=0)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for idx, img in enumerate(images):
            i, j = divmod(idx, grid_size)
            axes[i, j].imshow(img.reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
        fig_path = os.path.join(self.config.FIGURES_DIR, "04_latent_grid.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Сетка латентного пространства сохранена: {fig_path}")

    def _noise_experiment(self, decoder, x_test, encoder):
        imgs = x_test[:10]
        latent_real = encoder.predict(imgs, verbose=0)

        noise_levels = [0.5, 1.0]
        fig, axes = plt.subplots(2 + len(noise_levels), 10, figsize=(12, 2 + len(noise_levels)*1.2))

        # Оригиналы
        for i in range(10):
            axes[0, i].imshow(imgs[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
        axes[0, 0].set_title("Original")

        # Реконструкция без шума
        reconstructed = decoder.predict(latent_real, verbose=0)
        for i in range(10):
            axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[1, 0].set_title("Reconstructed (no noise)")

        # С шумом
        for row, noise_std in enumerate(noise_levels, start=2):
            noisy_latent = latent_real + np.random.normal(0, noise_std, latent_real.shape)
            noisy_images = decoder.predict(noisy_latent, verbose=0)
            for i in range(10):
                axes[row, i].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
                axes[row, i].axis('off')
            axes[row, 0].set_title(f"Noise σ={noise_std}")

        fig_path = os.path.join(self.config.FIGURES_DIR, "04_noise_experiment.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Эксперимент с шумом сохранён: {fig_path}")