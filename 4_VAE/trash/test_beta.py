import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

latent_dim = 16
batch_size = 256
epochs = 30

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * epsilon

class VAELossLayer(layers.Layer):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
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
        loss = recon + self.beta * kl
        self.add_loss(loss)
        return x_decoded

def build_and_train(beta_value, label):
    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(input_img)
    x = layers.Dense(128, activation='relu')(x)
    mu = layers.Dense(latent_dim)(x)
    log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    decoder_input = layers.Input(shape=(latent_dim,))
    decoder_h = layers.Dense(128, activation='relu')(decoder_input)
    decoder_output = layers.Dense(28*28, activation='sigmoid')(decoder_h)
    decoder_output = layers.Reshape((28, 28, 1))(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    decoded = decoder(z)
    decoded = VAELossLayer(beta=beta_value)([input_img, decoded, mu, log_var])
    vae = Model(input_img, decoded)
    vae.compile(optimizer='adam')

    vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.1, verbose=0)
    rec = vae.predict(x_test[:10], verbose=0)
    # Сохраняем картинку
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i+11)
        plt.imshow(rec[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f"VAE beta={beta_value}")
    plt.savefig(f"test_vae_beta{beta_value}.png", dpi=150)
    plt.close()
    print(f"Beta={beta_value}: final loss = {vae.evaluate(x_test, x_test, verbose=0):.4f}")

# Тестируем несколько значений
for beta in [0.001, 0.01, 0.1, 0.5, 1.0]:
    build_and_train(beta, str(beta))