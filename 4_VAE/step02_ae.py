import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from pipeline import PipelineStep
from sklearn.metrics import mean_squared_error

class AEStep(PipelineStep):
    def __init__(self, config, force=False):
        super().__init__("02_ae", config, force)
        self.model_path = os.path.join(config.CACHE_DIR, "ae_model.keras")

    def run(self):
        # Загружаем данные из шага 1
        x_train = np.load(os.path.join(self.config.CACHE_DIR, "x_train.npy"))
        x_test  = np.load(os.path.join(self.config.CACHE_DIR, "x_test.npy"))

        # Улучшенная архитектура: 256 → 128 → 64 → 128 → 256 → 784
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        encoded = Dense(64, activation='relu')(x)       # bottleneck 64
        x = Dense(128, activation='relu')(encoded)
        x = Dense(256, activation='relu')(x)
        decoded = Dense(28*28, activation='sigmoid')(x)
        decoded = Reshape((28, 28, 1))(decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        history = autoencoder.fit(
            x_train, x_train,
            epochs=self.config.EPOCHS_AE,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            verbose=1
        )

        autoencoder.save(self.model_path)
        print(f"Модель сохранена: {self.model_path}")

        # Визуализация реконструкции на тестовых примерах
        reconstructed = autoencoder.predict(x_test[:10], verbose=0)
        self._visualize(x_test[:10], reconstructed)

        # Вычисляем MSE на всём тестовом наборе для сравнения
        test_reconstructed = autoencoder.predict(x_test, verbose=0)
        x_test_flat = x_test.reshape(len(x_test), -1)
        test_reconstructed_flat = test_reconstructed.reshape(len(test_reconstructed), -1)
        mse_test = mean_squared_error(x_test_flat, test_reconstructed_flat)

        # Сохраняем последние BCE (для кривых обучения)
        loss_train = history.history['loss'][-1]
        loss_val   = history.history['val_loss'][-1] if 'val_loss' in history.history else None

        return {
            "model_path": self.model_path,
            "history": history.history,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "mse_test": mse_test,
            "test_reconstructed_flat": test_reconstructed_flat   # пригодится для сравнения
        }

    def _visualize(self, originals, reconstructed):
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_title("Originals")
        axes[1, 0].set_title("AE Reconstructed")
        plt.tight_layout()
        fig_path = self.config.FIGURES_DIR + "/02_ae_reconstruction.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Визуализация реконструкций: {fig_path}")