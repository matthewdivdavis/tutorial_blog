import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# 0. SETTINGS & PATHS
# ---------------------------------------------------------------------------
IMG_SIZE = 16          # images are IMG_SIZE x IMG_SIZE
RADIUS = 2.5           # circle radius
NOISE_DIM = 32
N_REAL_SAMPLES = 2000  # how many synthetic "real" images to generate
EPOCHS = 5000
SHAPE = "star"         # "star" or "circle"

DATA_SEED = 0                  # fixed, so the dataset is identical every render
TRAIN_SEED = int(time.time())  # set to a fixed int for reproducible training
torch.manual_seed(TRAIN_SEED)
print(f"Training seed: {TRAIN_SEED}")

# All paths are relative to the folder the .qmd / script lives in
ROOT = Path(".")
DATA_DIR = ROOT / "data"
SAMPLE_DIR = ROOT / "samples"
SNAPSHOT_DIR = ROOT / "snapshots"
G_PATH = ROOT / f"{SHAPE}_generator.pt"

for d in (DATA_DIR, SAMPLE_DIR, SNAPSHOT_DIR):
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DATA: synthetic images of a single shape at a random spot
# ---------------------------------------------------------------------------

def make_star_images(n, img_size=IMG_SIZE, outer_r=6.0, inner_r=2.4, n_points=5):
    """Returns (n, img_size, img_size) tensor, values in [-1, 1].
    Each image contains one 5-pointed star at a random position and rotation,
    rasterized with a point-in-polygon test.
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
    px = xs.reshape(-1).float()
    py = ys.reshape(-1).float()

    # Ray-casting point-in-polygon test, vectorized over batch and pixels
    inside = torch.zeros(n, px.shape[0], dtype=torch.bool)
    for i in range(v):
        j = (i + 1) % v
        x1, y1 = vx[:, i:i + 1], vy[:, i:i + 1]
        x2, y2 = vx[:, j:j + 1], vy[:, j:j + 1]
        cond_y = (y1 > py) != (y2 > py)
        x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
        inside ^= cond_y & (px < x_intersect)

    imgs = inside.float().view(n, img_size, img_size)
    return imgs * 2 - 1


MAKERS = {"star": make_star_images}


def load_or_make(name, n, data_seed=DATA_SEED):
    """Load a cached dataset if it exists, otherwise generate and save it.
    Stored as bool (images are binary) to keep the file small.
    """
    path = DATA_DIR / f"{name}_n{n}_seed{data_seed}.pt"

    if path.exists():
        mask = torch.load(path)
        print(f"Loaded cached dataset from {path}")
    else:
        # Fixed seed for the data only, without disturbing the training RNG
        with torch.random.fork_rng():
            torch.manual_seed(data_seed)
            imgs = MAKERS[name](n)
        mask = imgs > 0
        torch.save(mask, path)
        print(f"Generated and saved dataset to {path}")

    return mask.float() * 2 - 1  # back to [-1, 1]


REAL_DATA = load_or_make(SHAPE, N_REAL_SAMPLES)  # (N, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# 2. IMAGE SAVING HELPERS
# ---------------------------------------------------------------------------
def to_uint8(imgs):
    """(-1,1) tensor -> uint8 numpy array in [0, 255]."""
    return ((imgs.detach().cpu().numpy() + 1) / 2 * 255).astype("uint8")


def save_samples(n=5):
    """Save a few real training images as individual PNGs."""
    for i, img in enumerate(to_uint8(REAL_DATA[:n])):
        Image.fromarray(img).resize((128, 128), Image.NEAREST).save(
            SAMPLE_DIR / f"{SHAPE}_{i}.png"
        )


def save_grid(imgs, path, nrow=5, pad=2, scale=8):
    """Save a batch of (-1,1)-range images as one PNG grid."""
    imgs = to_uint8(imgs)
    n = imgs.shape[0]
    nrows_grid = -(-n // nrow)  # ceil
    cell = IMG_SIZE + pad
    grid = np.zeros((nrows_grid * cell, nrow * cell), dtype="uint8")
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        grid[r * cell: r * cell + IMG_SIZE, c * cell: c * cell + IMG_SIZE] = im
    Image.fromarray(grid).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
    ).save(path)


# ---------------------------------------------------------------------------
# 3. MODELS (plain MLPs)
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
# 4. TRAINING
# ---------------------------------------------------------------------------
def train(epochs=EPOCHS, batch_size=64, lr=2e-4, print_every=500,
          snapshot_dir=SNAPSHOT_DIR):
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(exist_ok=True)

    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(20, NOISE_DIM)  # same noise each snapshot, so you can watch it evolve

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
                samples = G(fixed_z)
            save_grid(samples, snapshot_dir / f"epoch_{epoch:05d}.png")

    return G


# ---------------------------------------------------------------------------
# 5. RUN (skips training if a saved generator already exists)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    save_samples()

    if G_PATH.exists():
        trained_G = Generator()
        trained_G.load_state_dict(torch.load(G_PATH))
        print(f"Loaded trained generator from {G_PATH}")
    else:
        trained_G = train()
        torch.save(trained_G.state_dict(), G_PATH)
        print(f"\nSaved generator to {G_PATH} and snapshots to {SNAPSHOT_DIR}/")

    # Final samples from the trained generator
    with torch.no_grad():
        save_grid(trained_G(torch.randn(20, NOISE_DIM)), ROOT / f"{SHAPE}_final.png")