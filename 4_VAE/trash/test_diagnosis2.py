import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

latent_dim = 16
batch_size = 256
epochs = 30

# Данные
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# Функция сэмплирования
def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon

# -------------------------------------------------
# Старый слой (reduce_mean) — из оригинального теста
# -------------------------------------------------
class VAELossLayerOld(layers.Layer):
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
        self.add_loss(loss)
        return x_decoded

# ------------------------------------------------
# Новый слой (reduce_sum / batch_size) — исправленный
# ------------------------------------------------
class VAELossLayerNew(layers.Layer):
    def call(self, inputs):
        x, x_decoded, mu, log_var = inputs
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        recon = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(
                tf.reshape(x, [-1, 28*28]),
                tf.reshape(x_decoded, [-1, 28*28])
            )
        ) / batch_size
        kl = tf.reduce_sum(-0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))) / batch_size
        loss = recon + kl
        self.add_loss(loss)
        return x_decoded

# ------------------------------------------------
# Тест 3 (старый слой, beta=1) — размытое среднее
# ------------------------------------------------
print("=== ТЕСТ 3 (старый слой, reduce_mean) ===")
input_img3 = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img3)
x = layers.Dense(128, activation='relu')(x)
mu3 = layers.Dense(latent_dim)(x)
log_var3 = layers.Dense(latent_dim)(x)
z3 = layers.Lambda(sampling, output_shape=(latent_dim,))([mu3, log_var3])

decoder_input3 = layers.Input(shape=(latent_dim,))
decoder_h3 = layers.Dense(128, activation='relu')(decoder_input3)
decoder_output3 = layers.Dense(28*28, activation='sigmoid')(decoder_h3)
decoder_output3 = layers.Reshape((28, 28, 1))(decoder_output3)
decoder3 = Model(decoder_input3, decoder_output3)

decoded3 = decoder3(z3)
decoded3 = VAELossLayerOld(beta=1.0)([input_img3, decoded3, mu3, log_var3])
vae_old = Model(input_img3, decoded3)
vae_old.compile(optimizer='adam')

history3 = vae_old.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=1)
rec_old = vae_old.predict(x_test[:10], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(rec_old[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Тест 3 (reduce_mean) – должно быть размыто")
plt.savefig("test_vae_old_mean.png", dpi=150)
plt.close()
print("Финальный loss (старый):", history3.history['loss'][-1])

# ---------------------------------------------------------
# Тест 4 (новый слой, reduce_sum / batch_size) – должно быть чётко
# ---------------------------------------------------------
print("\n=== ТЕСТ 4 (новый слой, reduce_sum / batch_size) ===")
input_img4 = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(input_img4)
x = layers.Dense(128, activation='relu')(x)
mu4 = layers.Dense(latent_dim)(x)
log_var4 = layers.Dense(latent_dim)(x)
z4 = layers.Lambda(sampling, output_shape=(latent_dim,))([mu4, log_var4])

decoder_input4 = layers.Input(shape=(latent_dim,))
decoder_h4 = layers.Dense(128, activation='relu')(decoder_input4)
decoder_output4 = layers.Dense(28*28, activation='sigmoid')(decoder_h4)
decoder_output4 = layers.Reshape((28, 28, 1))(decoder_output4)
decoder4 = Model(decoder_input4, decoder_output4)

decoded4 = decoder4(z4)
decoded4 = VAELossLayerNew()([input_img4, decoded4, mu4, log_var4])
vae_new = Model(input_img4, decoded4)
vae_new.compile(optimizer='adam')

history4 = vae_new.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=1)
rec_new = vae_new.predict(x_test[:10], verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(rec_new[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Тест 4 (reduce_sum/batch) – должно быть чётко")
plt.savefig("test_vae_new_sum.png", dpi=150)
plt.close()
print("Финальный loss (новый):", history4.history['loss'][-1])