import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from pipeline import PipelineStep
from step03_vae import sampling   # для загрузки VAE


class CompareStep(PipelineStep):
    def __init__(self, config, force=False, latent_dim=None):
        if latent_dim is None:
            latent_dim = config.LATENT_DIMS[0]
        super().__init__(f"05_compare_dim{latent_dim}", config, force)
        self.latent_dim = latent_dim
        self.ae_model_path = os.path.join(config.CACHE_DIR, "ae_model.keras")
        self.vae_full_path = os.path.join(config.CACHE_DIR, f"vae_full_dim{latent_dim}.keras")

    def run(self):
        # Загружаем данные
        x_train = np.load(os.path.join(self.config.CACHE_DIR, "x_train.npy"))
        x_test  = np.load(os.path.join(self.config.CACHE_DIR, "x_test.npy"))
        y_train = np.load(os.path.join(self.config.CACHE_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(self.config.CACHE_DIR, "y_test.npy"))

        x_train_flat = x_train.reshape(len(x_train), -1)
        x_test_flat  = x_test.reshape(len(x_test), -1)

        # Загружаем модели
        ae = tf.keras.models.load_model(self.ae_model_path, compile=False)
        vae = tf.keras.models.load_model(self.vae_full_path, compile=False,
                                         custom_objects={'sampling': sampling})

        # PCA
        pca64 = PCA(n_components=64)
        pca64.fit(x_train_flat)
        x_test_pca64_recon = pca64.inverse_transform(pca64.transform(x_test_flat))

        # Предсказания AE
        ae_recon = ae.predict(x_test, verbose=0)
        ae_recon_flat = ae_recon.reshape(len(ae_recon), -1)

        # Предсказания VAE (реконструкция)
        vae_recon = vae.predict(x_test, verbose=0)
        vae_recon_flat = vae_recon.reshape(len(vae_recon), -1)

        # MSE
        mse_pca = mean_squared_error(x_test_flat, x_test_pca64_recon)
        mse_ae  = mean_squared_error(x_test_flat, ae_recon_flat)
        mse_vae = mean_squared_error(x_test_flat, vae_recon_flat)

        print(f'MSE PCA  (64 comp): {mse_pca:.4f}')
        print(f'MSE AE   (64 dim):  {mse_ae:.4f}')
        print(f'MSE VAE  ({self.latent_dim} dim):   {mse_vae:.4f}')

        # 1. Сравнение реконструкций (визуально)
        self._plot_reconstructions(x_test_flat, x_test_pca64_recon, ae_recon_flat, vae_recon_flat)

        # 2. Бар-чарт MSE
        self._plot_mse_bars(mse_pca, mse_ae, mse_vae)

        # 3. Латентное пространство PCA vs VAE (только если VAE dim=2)
        if self.latent_dim == 2:
            self._plot_latent_spaces(pca64, x_train_flat, y_train, vae, x_test)

        # 4. График объяснённой дисперсии PCA
        self._plot_explained_variance(pca64, x_train_flat)

        # Итоговая сводка
        result = {
            "mse_pca": mse_pca,
            "mse_ae": mse_ae,
            "mse_vae": mse_vae,
        }
        return result

    def _plot_reconstructions(self, x_test_flat, pca_recon, ae_recon, vae_recon):
        n = 10
        fig, axes = plt.subplots(4, n, figsize=(14, 6))
        rows_data = [x_test_flat[:n], pca_recon[:n], ae_recon[:n], vae_recon[:n]]
        row_names = ['Оригинал', 'PCA', 'AE', 'VAE']

        for row_idx, (name, data) in enumerate(zip(row_names, rows_data)):
            for col_idx in range(n):
                axes[row_idx, col_idx].imshow(data[col_idx].reshape(28, 28), cmap='gray')
                axes[row_idx, col_idx].axis('off')
            axes[row_idx, 0].set_ylabel(name, fontsize=9)

        plt.suptitle('Сравнение реконструкций: PCA vs AE vs VAE', fontsize=12)
        plt.tight_layout()
        fig_path = self.config.FIGURES_DIR + "/05_reconstructions_comparison.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Сравнение реконструкций сохранено: {fig_path}")

    def _plot_mse_bars(self, mse_pca, mse_ae, mse_vae):
        methods = [f'PCA\n64d', f'AE\n64d', f'VAE\n{self.latent_dim}d']
        mses = [mse_pca, mse_ae, mse_vae]
        colors = ['#4878CF', '#6ACC65', '#D65F5F']

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(methods, mses, color=colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.0001, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=11)
        ax.set_ylabel('MSE')
        ax.set_title('Ошибка реконструкции: PCA vs AE vs VAE')
        ax.grid(True, axis='y', alpha=0.3)
        fig_path = self.config.FIGURES_DIR + "/05_mse_comparison.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"График MSE сохранён: {fig_path}")

    def _plot_latent_spaces(self, pca64, x_train_flat, y_train, vae, x_test):
        # PCA 2D
        pca2 = PCA(n_components=2)
        x_train_pca2 = pca2.fit_transform(x_train_flat)

        # VAE mu для тренировочных (возьмем первые 5000)
        # Загружаем энкодер отдельно или используем функциональную часть vae
        # Проще: используем vae.predict для получения mu? Но модель vae возвращает только реконструкцию.
        # Нужна модель до mu. Можно загрузить vae_encoder_dim{latent_dim}.keras, но здесь у нас только full.
        # Поэтому загрузим энкодер явно.
        encoder_path = os.path.join(self.config.CACHE_DIR, f"vae_encoder_dim{self.latent_dim}.keras")
        encoder = tf.keras.models.load_model(encoder_path, compile=False,
                                            custom_objects={'sampling': sampling})
        mu_train = []
        with tf.device('/cpu:0'):  # не нужно GPU для инференса 5000 примеров
            for i in range(0, min(5000, len(x_train_flat)), 256):
                batch = x_train_flat[i:i+256].reshape(-1, 28, 28, 1)
                mu_batch = encoder.predict(batch, verbose=0)
                mu_train.append(mu_batch)
        mu_train = np.concatenate(mu_train)[:5000]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sc = axes[0].scatter(x_train_pca2[:5000, 0], x_train_pca2[:5000, 1],
                             c=y_train[:5000], cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(sc, ax=axes[0], ticks=range(10))
        axes[0].set_title('Латентное пространство PCA (2D)')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')

        sc2 = axes[1].scatter(mu_train[:, 0], mu_train[:, 1],
                              c=y_train[:5000], cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(sc2, ax=axes[1], ticks=range(10))
        axes[1].set_title(f'Латентное пространство VAE (2D, μ)')
        axes[1].set_xlabel('z[0]')
        axes[1].set_ylabel('z[1]')

        plt.suptitle('PCA vs VAE: структура латентного пространства', fontsize=12)
        fig_path = self.config.FIGURES_DIR + "/05_latent_space_comparison.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Сравнение латентных пространств: {fig_path}")

    def _plot_explained_variance(self, pca64, x_train_flat):
        cumsum = np.cumsum(pca64.explained_variance_ratio_)
        components_range = list(range(1, 65))

        plt.figure(figsize=(9, 5))
        plt.plot(components_range, cumsum, marker='o', markersize=4, color='steelblue')
        plt.axhline(cumsum[-1], linestyle='--', color='orange',
                    label=f'Максимум при 64 компонентах ({cumsum[-1]*100:.1f}%)')
        plt.xlabel('Число компонент')
        plt.ylabel('Накопленная доля дисперсии')
        plt.title('PCA: объяснённая дисперсия vs число компонент')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_path = self.config.FIGURES_DIR + "/05_pca_explained_variance.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"График дисперсии PCA сохранён: {fig_path}")