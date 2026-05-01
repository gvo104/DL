import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from pipeline import PipelineStep

class AEStep(PipelineStep):
    def __init__(self, config, force=False):
        super().__init__("02_ae", config, force)
        self.model_path = os.path.join(config.CACHE_DIR, "ae_model.keras")

    def run(self):
        # Загружаем данные (они уже должны быть подготовлены)
        x_train = np.load(os.path.join(self.config.CACHE_DIR, "x_train.npy"))
        x_test  = np.load(os.path.join(self.config.CACHE_DIR, "x_test.npy"))

        # Архитектура AE
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        encoded = Dense(64, activation='relu')(x)
        decoded = Dense(28 * 28, activation='sigmoid')(encoded)
        decoded = Reshape((28, 28, 1))(decoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Обучение
        history = autoencoder.fit(
            x_train, x_train,
            epochs=self.config.EPOCHS_AE,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            verbose=1
        )

        # Сохраняем модель
        autoencoder.save(self.model_path)
        print(f"Модель сохранена: {self.model_path}")

        # Визуализация реконструкции на тестовых примерах
        reconstructed = autoencoder.predict(x_test[:10], verbose=0)
        self._visualize(x_test[:10], reconstructed)

        # Собираем метрики
        mse_train = history.history['loss'][-1]
        mse_val   = history.history['val_loss'][-1] if 'val_loss' in history.history else None

        return {
            "model_path": self.model_path,
            "history": history.history,
            "mse_train": mse_train,
            "mse_val": mse_val
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