import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

latent_dim = 16
epochs = 30
batch_size = 256

# ------------------------------
# 1. Данные
# ------------------------------
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# ------------------------------
# 2. Архитектура
# ------------------------------
input_img = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img)
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

decoded = decoder(z)

# ------------------------------
# 3. Слой потерь (правильный способ)
# ------------------------------
class VAELossLayer(layers.Layer):
    def call(self, inputs):
        x, x_decoded, mu, log_var = inputs
        # Средняя кросс-энтропия
        recon = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.reshape(x, [tf.shape(x)[0], -1]),
                tf.reshape(x_decoded, [tf.shape(x_decoded)[0], -1])
            )
        )
        # Средний KL
        kl = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        loss = recon + kl
        self.add_loss(loss)
        return x_decoded

decoded = VAELossLayer()([input_img, decoded, mu, log_var])

vae = Model(input_img, decoded)
vae.compile(optimizer='adam')

# ------------------------------
# 4. Обучение
# ------------------------------
history = vae.fit(x_train, x_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  verbose=1)

# ------------------------------
# 5. Визуализация реконструкции
# ------------------------------
reconstructed = vae.predict(x_test[:10], verbose=0)
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.suptitle("Original vs VAE Reconstruction")
plt.savefig("test_vae_reconstruction.png", dpi=150)
plt.close()
print("Готово! Откройте test_vae_reconstruction.png")