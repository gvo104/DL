import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

latent_dim = 16
batch_size = 256
epochs = 30

# ------------------------------
# Данные
# ------------------------------
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# ============================================================
# ТЕСТ 1: Обычный AE (без VAE) – проверка архитектуры
# ============================================================
print("=== ТЕСТ 1: Обычный AE ===")
input_img = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img)
x = layers.Dense(128, activation='relu')(x)
bottleneck = layers.Dense(latent_dim)(x)
x = layers.Dense(128, activation='relu')(bottleneck)
x = layers.Dense(28*28, activation='sigmoid')(x)
decoded = layers.Reshape((28, 28, 1))(x)
ae = Model(input_img, decoded)
ae.compile(optimizer='adam', loss='binary_crossentropy')
ae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
rec_ae = ae.predict(x_test[:10], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(rec_ae[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("AE (bottleneck=16) – должно быть чётко")
plt.savefig("test_ae_recon.png", dpi=150)
plt.close()
print("Готово. Откройте test_ae_recon.png")
# Если здесь размыто – проблема в архитектуре/данных/оптимизации.
# Если чётко – идём дальше.

# ============================================================
# ТЕСТ 2: VAE с отключенным KL (beta=0) – должен работать как AE
# ============================================================
print("\n=== ТЕСТ 2: VAE с beta=0 (без KL) ===")
input_img2 = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img2)
x = layers.Dense(128, activation='relu')(x)
mu = layers.Dense(latent_dim)(x)
log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

# Декодер
decoder_input = layers.Input(shape=(latent_dim,))
decoder_h = layers.Dense(128, activation='relu')(decoder_input)
decoder_output = layers.Dense(28*28, activation='sigmoid')(decoder_h)
decoder_output = layers.Reshape((28, 28, 1))(decoder_output)
decoder = Model(decoder_input, decoder_output)

decoded2 = decoder(z)

# Слой потерь с beta=0
class VAELossLayerBeta(layers.Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    def call(self, inputs):
        x, x_decoded, mu, log_var = inputs
        recon = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.reshape(x, [tf.shape(x)[0], -1]),
                tf.reshape(x_decoded, [tf.shape(x_decoded)[0], -1])
            )
        )
        kl = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        loss = recon + self.beta * kl
        # Для диагностики будем выводить значения
        self.add_loss(loss)
        # Сохраним последние значения для печати
        self._last_recon = recon
        self._last_kl = kl
        return x_decoded

loss_layer = VAELossLayerBeta(beta=0.0)
decoded2 = loss_layer([input_img2, decoded2, mu, log_var])

vae_no_kl = Model(input_img2, decoded2)
vae_no_kl.compile(optimizer='adam')

# Коллбек для вывода значений потерь
class LossPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        recon = self.model.layers[-1]._last_recon.numpy()
        kl = self.model.layers[-1]._last_kl.numpy()
        print(f"  recon={recon:.4f}, kl={kl:.4f}")

vae_no_kl.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=0, callbacks=[LossPrinter()])
rec_vae_nokl = vae_no_kl.predict(x_test[:10], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(rec_vae_nokl[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("VAE (beta=0) – должно быть как AE")
plt.savefig("test_vae_beta0.png", dpi=150)
plt.close()
print("Готово. Откройте test_vae_beta0.png (должно быть чётко)")

# ============================================================
# ТЕСТ 3: Полный VAE (beta=1) с мониторингом потерь
# ============================================================
print("\n=== ТЕСТ 3: Полный VAE (beta=1) ===")
input_img3 = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img3)
x = layers.Dense(128, activation='relu')(x)
mu3 = layers.Dense(latent_dim)(x)
log_var3 = layers.Dense(latent_dim)(x)

z3 = layers.Lambda(sampling, output_shape=(latent_dim,))([mu3, log_var3])

decoder3_input = layers.Input(shape=(latent_dim,))
decoder3_h = layers.Dense(128, activation='relu')(decoder3_input)
decoder3_output = layers.Dense(28*28, activation='sigmoid')(decoder3_h)
decoder3_output = layers.Reshape((28, 28, 1))(decoder3_output)
decoder3 = Model(decoder3_input, decoder3_output)

decoded3 = decoder3(z3)

loss_layer3 = VAELossLayerBeta(beta=1.0)
decoded3 = loss_layer3([input_img3, decoded3, mu3, log_var3])

vae_full = Model(input_img3, decoded3)
vae_full.compile(optimizer='adam')

vae_full.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
             validation_split=0.1, verbose=0, callbacks=[LossPrinter()])
rec_vae_full = vae_full.predict(x_test[:10], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(rec_vae_full[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("VAE (beta=1) – реконструкция")
plt.savefig("test_vae_beta1.png", dpi=150)
plt.close()
print("Готово. Откройте test_vae_beta1.png")

print("\nВсе тесты завершены. Анализируйте изображения и консольный вывод.")