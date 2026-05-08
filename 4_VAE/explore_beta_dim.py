import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

LATENT_DIMS = [2, 
               #8, 16
               ]
BETAS = [
    #0.0,
         0.0005, 
         ]
EPOCHS = 30
BATCH_SIZE = 256

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28, 28, 1)
x_test  = (x_test.astype('float32')  / 255.0).reshape(-1, 28, 28, 1)

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
        kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var)) / batch_size
        loss = recon + self.beta * kl
        self.add_loss(loss)
        return x_decoded

def build_vae(latent_dim, beta):
    input_img = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(input_img)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    mu = layers.Dense(latent_dim)(x)
    log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

    decoder_input = layers.Input(shape=(latent_dim,))
    decoder_h = layers.Dense(128, activation='relu')(decoder_input)
    decoder_h = layers.Dense(256, activation='relu')(decoder_h)
    decoder_output = layers.Dense(28*28, activation='sigmoid')(decoder_h)
    decoder_output = layers.Reshape((28, 28, 1))(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    decoded = decoder(z)
    decoded = VAELossLayer(beta=beta)([input_img, decoded, mu, log_var])

    vae = Model(input_img, decoded)
    mu_model = Model(input_img, mu)
    return vae, decoder, mu_model

results = {}
for latent_dim in LATENT_DIMS:
    print(f"\n========== LATENT DIM = {latent_dim} ==========")
    for beta in BETAS:
        print(f"Training dim={latent_dim}, beta={beta:.4f} ...")
        vae, decoder, mu_model = build_vae(latent_dim, beta)
        vae.compile(optimizer='adam')
        history = vae.fit(x_train, x_train,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         validation_split=0.1,
                         verbose=0)
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
        test_loss = vae.evaluate(x_test, x_test, verbose=0)

        print(f"  -> beta={beta:.4f}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")

        z_sample = np.random.normal(0, 1, size=(10, latent_dim))
        generated = decoder.predict(z_sample, verbose=0)
        mu_test = mu_model.predict(x_test, verbose=0)

        results[(latent_dim, beta)] = {
            'generated': generated,
            'mu_test': mu_test,
            'y_test': y_test,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss
        }

# Визуализация (как раньше, с реальными образцами сверху и генерацией снизу)
real_samples = x_test[:10]
for latent_dim in LATENT_DIMS:
    n_betas = len(BETAS)
    fig, axes = plt.subplots(n_betas, 2, figsize=(12, 2.8 * n_betas))
    if n_betas == 1:
        axes = axes.reshape(1, -1)
    for row, beta in enumerate(BETAS):
        data = results[(latent_dim, beta)]
        gen_imgs = data['generated']
        mu_test = data['mu_test']
        labels = data['y_test']
        train_loss = data['train_loss']
        test_loss = data['test_loss']

        ax_left = axes[row, 0]
        top_row = np.hstack([real_samples[i].reshape(28, 28) for i in range(10)])
        bottom_row = np.hstack([gen_imgs[i].reshape(28, 28) for i in range(10)])
        combined = np.vstack([top_row, bottom_row])
        ax_left.imshow(combined, cmap='gray')
        ax_left.axis('off')
        ax_left.set_title(f"β={beta:.4f}, train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

        ax_right = axes[row, 1]
        if latent_dim > 2:
            mu_2d = PCA(n_components=2).fit_transform(mu_test)
        else:
            mu_2d = mu_test
        scatter = ax_right.scatter(mu_2d[:, 0], mu_2d[:, 1],
                                   c=labels, cmap='tab10', s=0.5, alpha=0.7)
        ax_right.set_title(f"PCA latent (β={beta:.4f})")
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        if row == n_betas - 1:
            plt.colorbar(scatter, ax=ax_right, ticks=range(10))

    plt.suptitle(f"Latent dimension = {latent_dim}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"explore_dim{latent_dim}.png", dpi=150)
    plt.close()
    print(f"Saved explore_dim{latent_dim}.png")

print("\nГотово!")