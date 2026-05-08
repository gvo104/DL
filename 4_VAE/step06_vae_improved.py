import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Improved VAE (β-VAE)
# =========================
class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =========================
# Loss with beta
# =========================
def loss_fn(recon, x, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl, bce, kl


# =========================
# Training
# =========================
def train(model, loader, device, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"bce": [], "kl": [], "beta": []}

    for epoch in range(cfg.epochs):
        model.train()

        # KL annealing
        beta = min(cfg.beta_max, epoch / cfg.anneal_epochs * cfg.beta_max)

        total_bce = 0
        total_kl = 0

        for imgs, _ in loader:
            x = imgs.view(imgs.size(0), -1).to(device)

            recon, mu, logvar = model(x)
            loss, bce, kl = loss_fn(recon, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += bce.item()
            total_kl += kl.item()

        avg_bce = total_bce / len(loader.dataset)
        avg_kl = total_kl / len(loader.dataset)

        history["bce"].append(avg_bce)
        history["kl"].append(avg_kl)
        history["beta"].append(beta)

        print(f"  epoch {epoch+1:02d} | BCE={avg_bce:.2f} | KL={avg_kl:.2f} | beta={beta:.3f}")

    return history


# =========================
# Visualization
# =========================
def plot_training(history, save_path):
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(history["bce"], label="BCE")
    ax1.plot(history["kl"], label="KL")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(history["beta"], linestyle="--", label="beta")
    ax2.set_ylabel("beta")

    plt.title("β-VAE training dynamics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_samples(model, device, latent_dim, save_path):
    model.eval()

    with torch.no_grad():
        z = torch.randn(25, latent_dim).to(device)
        imgs = model.decode(z).cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# Run step
# =========================
def run(cfg, data, device):
    print("\nStep 06: improved VAE (β-VAE + annealing)")

    model = BetaVAE(latent_dim=cfg.latent_dim_improved).to(device)

    history = train(model, data["train_loader"], device, cfg)

    # plots
    plot_training(history, cfg.figures_dir / "06_training.png")
    generate_samples(model, device, cfg.latent_dim_improved,
                     cfg.figures_dir / "06_generation.png")

    return {"model": model, "history": history}