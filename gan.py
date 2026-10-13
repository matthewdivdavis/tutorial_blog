import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from PIL import Image
from pathlib import Path

seed = time.time()
torch.manual_seed(seed)

# torch.manual_seed(0)

IMG_SIZE = 16          # images are IMG_SIZE x IMG_SIZE
RADIUS = 2.5
NOISE_DIM = 32
N_REAL_SAMPLES = 2000  # how many synthetic "real" images to generate
EPOCHS = 5000

ROOT = Path(".")                  # folder containing the .qmd
DATA_DIR = ROOT / "data"
SAMPLE_DIR = ROOT / "samples"
SNAPSHOT_DIR = ROOT / "snapshots"
for d in (DATA_DIR, SAMPLE_DIR, SNAPSHOT_DIR):
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DATA: synthetic images, each a single filled circle at a random spot
# ---------------------------------------------------------------------------
def load_or_make(name, make_fn, n, data_seed=0):
    """Load a cached dataset if it exists, otherwise generate and save it."""
    path = DATA_DIR / f"{name}_n{n}_seed{data_seed}.pt"

    if path.exists():
        mask = torch.load(path)
        print(f"Loaded cached dataset from {path}")
    else:
        # Use a fixed seed for the data only, without disturbing the global RNG
        with torch.random.fork_rng():
            torch.manual_seed(data_seed)
            imgs = make_fn(n)
        mask = imgs > 0                      # {-1, 1} -> {False, True}
        torch.save(mask, path)
        print(f"Generated and saved dataset to {path}")

    return mask.float() * 2 - 1              # back to [-1, 1]


REAL_DATA = load_or_make("star", make_star_images, N_REAL_SAMPLES)
# REAL_DATA = load_or_make("circle", make_circle_images, N_REAL_SAMPLES)

def make_star_images(n, img_size=16, outer_r=6.0, inner_r=2.4, n_points=5):
    """Returns (n, img_size, img_size) tensor, values in [-1, 1].
    Each image contains one 5-pointed star at a random position and rotation.
    Same idea as make_circle_images: rasterize a shape via a per-pixel test,
    here point-in-polygon instead of distance-to-center.
    """
    margin = outer_r + 1
    cx = torch.empty(n).uniform_(margin, img_size - margin)
    cy = torch.empty(n).uniform_(margin, img_size - margin)
    rotation = torch.empty(n).uniform_(0, 2 * math.pi)
 
    # Star polygon: 2*n_points vertices alternating outer/inner radius
    v = 2 * n_points
    k = torch.arange(v)
    radii = torch.where(k % 2 == 0, outer_r, inner_r).float()          # (v,)
    angles = rotation.unsqueeze(1) + k.float() * (math.pi / n_points)  # (n, v)
    vx = cx.unsqueeze(1) + radii * torch.cos(angles)                   # (n, v)
    vy = cy.unsqueeze(1) + radii * torch.sin(angles)                   # (n, v)
 
    # Pixel grid, flattened to (P,)
    ys, xs = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij")
    px = xs.reshape(-1).float()  # (P,)
    py = ys.reshape(-1).float()  # (P,)
 
    # Ray-casting point-in-polygon test, vectorized over batch (n) and pixels (P)
    inside = torch.zeros(n, px.shape[0], dtype=torch.bool)
    for i in range(v):
        x1, y1 = vx[:, i:i + 1], vy[:, i:i + 1]                # (n, 1)
        x2, y2 = vx[:, (i + 1) % v:(i + 1) % v + 1], vy[:, (i + 1) % v:(i + 1) % v + 1]
        cond_y = (y1 > py) != (y2 > py)                        # (n, P)
        x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
        inside ^= cond_y & (px < x_intersect)
 
    imgs = inside.float().view(n, img_size, img_size)
    return imgs * 2 - 1  # rescale {0,1} -> {-1,1}, matches Tanh output range


# REAL_DATA = make_circle_images(N_REAL_SAMPLES)  # (N, IMG_SIZE, IMG_SIZE)
REAL_DATA = make_star_images(N_REAL_SAMPLES)

def save_samples():
    for i, img in enumerate(REAL_DATA[0:5]):
        arr = ((img.numpy() + 1) / 2 * 255).astype("uint8")  # [-1,1] -> [0,255]
        Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(f"/Users/matthewdavis/Desktop/Programming_stuff/Python/Personal/GAN/samples/star_{i}.png")


def save_grid(imgs, path, nrow=5):
    """Save a batch of (-1,1)-range images as one PNG grid, no extra deps."""
    import numpy as np
    from PIL import Image

    imgs = ((imgs.detach().cpu().numpy() + 1) / 2 * 255).astype("uint8")
    n = imgs.shape[0]
    ncol = -(-n // nrow)  # ceil
    pad = 2
    grid = np.zeros((ncol * (IMG_SIZE + pad), nrow * (IMG_SIZE + pad)), dtype="uint8")
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        grid[
            r * (IMG_SIZE + pad): r * (IMG_SIZE + pad) + IMG_SIZE,
            c * (IMG_SIZE + pad): c * (IMG_SIZE + pad) + IMG_SIZE,
        ] = im
    Image.fromarray(grid).resize((grid.shape[1] * 8, grid.shape[0] * 8), Image.NEAREST).save(path)


# ---------------------------------------------------------------------------
# 2. MODELS (same shape of architecture as the name GAN: plain MLPs)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, IMG_SIZE * IMG_SIZE),
            nn.Tanh(),  # output in [-1, 1], same range as the real data
        )

    def forward(self, z):
        return self.net(z).view(-1, IMG_SIZE, IMG_SIZE)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),  # raw logit
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# 3. TRAINING (identical loop/losses to the name GAN)
# ---------------------------------------------------------------------------
def train(epochs=EPOCHS, batch_size=64, lr=2e-4, print_every=500, snapshot_dir="/Users/matthewdavis/Desktop/Programming_stuff/Python/Personal/GAN/snapshots/"):
    import os
    os.makedirs(snapshot_dir, exist_ok=True)

    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, len(REAL_DATA), (batch_size,))
        real = REAL_DATA[idx]

        z = torch.randn(batch_size, NOISE_DIM)
        fake = G(z)

        # === Train Discriminator ===
        opt_d.zero_grad()
        real_pred = D(real)
        fake_pred = D(fake.detach())
        d_loss = loss_fn(real_pred, torch.full_like(real_pred, 0.9)) + \
            loss_fn(fake_pred, torch.zeros_like(fake_pred))
        d_loss.backward()
        opt_d.step()

        # === Train Generator ===
        opt_g.zero_grad()
        fake_pred = D(fake)
        g_loss = loss_fn(fake_pred, torch.ones_like(fake_pred))
        g_loss.backward()
        opt_g.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | D loss {d_loss.item():.3f} | G loss {g_loss.item():.3f}")
            with torch.no_grad():
                z = torch.randn(20, NOISE_DIM)
                samples = G(z)
            save_grid(samples, f"{snapshot_dir}/epoch_{epoch:05d}.png")

    return G


if __name__ == "__main__":
    save_samples()
    trained_G = train()
    print("\nSaved training snapshots to /Users/matthewdavis/Desktop/Programming_stuff/Python/Personal/GAN/snapshots/")
    torch.save(trained_G.state_dict(), "/Users/matthewdavis/Desktop/Programming_stuff/Python/Personal/GAN/image_generator.py")