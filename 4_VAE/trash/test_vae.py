import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

latent_dim = 2
epochs = 30
batch_size = 256

# Данные
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# Архитектура
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

# Потери вручную через add_loss
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(
        tf.reshape(input_img, [-1, 28*28]),
        tf.reshape(decoded, [-1, 28*28])
    )
)
kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))

vae = Model(input_img, decoded)
vae.add_loss(reconstruction_loss + kl_loss)        # <-- добавляем прямо здесь
vae.compile(optimizer='adam')

history = vae.fit(x_train, x_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  verbose=1)

# Проверка реконструкции
reconstructed = vae.predict(x_test[:10], verbose=0)
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.suptitle("Original (top) vs VAE Reconstruction (bottom)")
plt.savefig("test_vae_reconstruction.png", dpi=150)
plt.close()
print("Готово! Смотрите test_vae_reconstruction.png")