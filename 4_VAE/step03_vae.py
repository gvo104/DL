import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Reshape,
                                     Lambda, Layer)
from tensorflow.keras import backend as K
from pipeline import PipelineStep


# Регистрируем функцию сэмплирования, чтобы Keras мог сериализовать Lambda-слой
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon


class VAELossLayer(Layer):
    """Слой, вычисляющий VAE loss и добавляющий его в модель."""
    def call(self, inputs):
        x, x_decoded, mu, log_var = inputs
        recon_loss = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.reshape(x, [tf.shape(x)[0], -1]),
                tf.reshape(x_decoded, [tf.shape(x_decoded)[0], -1])
            )
        )
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var))
        loss = recon_loss + kl_loss
        self.add_loss(loss)
        return x_decoded


class VAEStep(PipelineStep):
    def __init__(self, config, force=False):
        super().__init__("03_vae", config, force)
        self.latent_dim = 2
        self.encoder_path = os.path.join(config.CACHE_DIR, "vae_encoder.keras")
        self.decoder_path = os.path.join(config.CACHE_DIR, "vae_decoder.keras")
        self.full_model_path = os.path.join(config.CACHE_DIR, "vae_full.keras")

    def run(self):
        # Загружаем данные
        x_train = np.load(os.path.join(self.config.CACHE_DIR, "x_train.npy"))
        x_test  = np.load(os.path.join(self.config.CACHE_DIR, "x_test.npy"))
        y_test  = np.load(os.path.join(self.config.CACHE_DIR, "y_test.npy"))

        # --- Архитектура VAE ---
        input_img = Input(shape=(28, 28, 1), name='input_img')
        x = Flatten()(input_img)
        x = Dense(128, activation='relu')(x)
        mu = Dense(self.latent_dim, name='mu')(x)
        log_var = Dense(self.latent_dim, name='log_var')(x)

        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([mu, log_var])

        # Декодер
        decoder_input = Input(shape=(self.latent_dim,))
        decoder_h = Dense(128, activation='relu')(decoder_input)
        decoder_output = Dense(28 * 28, activation='sigmoid')(decoder_h)
        decoder_output = Reshape((28, 28, 1))(decoder_output)
        decoder = Model(decoder_input, decoder_output, name='decoder')

        decoded = decoder(z)
        decoded = VAELossLayer()([input_img, decoded, mu, log_var])

        vae = Model(input_img, decoded)
        vae.compile(optimizer='adam')

        # Обучение
        history = vae.fit(
            x_train, x_train,
            epochs=self.config.EPOCHS_VAE,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            verbose=1
        )

        # Сохраняем модели
        vae.save(self.full_model_path)

        encoder = Model(input_img, z, name='encoder')
        encoder.save(self.encoder_path)
        decoder.save(self.decoder_path)

        # Реконструкция
        reconstructed = vae.predict(x_test[:10], verbose=0)
        self._visualize_reconstruction(x_test[:10], reconstructed)

        # Латентное пространство (mu)
        mu_model = Model(input_img, mu, name='mu_model')
        mu_test = mu_model.predict(x_test, verbose=0)
        self._visualize_latent_space(mu_test, y_test)

        loss_train = history.history['loss'][-1]
        loss_val = history.history['val_loss'][-1] if 'val_loss' in history.history else None

        return {
            "encoder_path": self.encoder_path,
            "decoder_path": self.decoder_path,
            "full_model_path": self.full_model_path,
            "history": history.history,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "latent_dim": self.latent_dim
        }

    def _visualize_reconstruction(self, originals, reconstructed):
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_title("Originals")
        axes[1, 0].set_title("VAE Reconstructed")
        plt.tight_layout()
        fig_path = self.config.FIGURES_DIR + "/03_vae_reconstruction.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Визуализация реконструкций VAE: {fig_path}")

    def _visualize_latent_space(self, mu, labels):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', s=1, alpha=0.7)
        plt.colorbar(scatter, ticks=range(10))
        plt.title("VAE Latent Space (mu)")
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        fig_path = self.config.FIGURES_DIR + "/03_vae_latent_space.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Визуализация латентного пространства VAE: {fig_path}")