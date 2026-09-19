import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

BATCH_SIZE = 32
EPOCHS = 8

LR = 2e-4
WEIGHT_DECAY = 1e-4

POS_WEIGHT = 1.0

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V7 DISTANCE-AWARE U-NET")
print("=" * 70)

print("\nUsing device:", DEVICE)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PATHS
# ============================================================

BASE = "data/processed/unet"

TRAIN_COORDS_PATH = f"{BASE}/train_coords.npy"
VAL_COORDS_PATH = f"{BASE}/val_coords.npy"

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = (
    "models/fault_v7_distance_unet_best.pt"
)


# ============================================================
# LOAD RASTER
# ============================================================

import rasterio

print("\nLoading feature raster...")

with rasterio.open(FEATURE_PATH) as src:
    features = src.read().astype(np.float32)

print("Features:", features.shape)

print("\nLoading labels...")

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

print("Labels:", labels.shape)


# ============================================================
# NORMALIZATION
# ============================================================

means = np.load(
    MEANS_PATH
).astype(np.float32)

stds = np.load(
    STDS_PATH
).astype(np.float32)

print("\nNormalizing features...")

for b in range(features.shape[0]):

    x = features[b]

    invalid = (
        ~np.isfinite(x)
        |
        (np.abs(x) > 1e30)
    )

    x[invalid] = means[b]

    x = (
        x - means[b]
    ) / (
        stds[b] + 1e-8
    )

    x = np.clip(
        x,
        -10.0,
        10.0
    )

    x[invalid] = 0.0

    features[b] = x


# ============================================================
# COORDINATES
# ============================================================

train_coords = np.load(
    TRAIN_COORDS_PATH
)

val_coords = np.load(
    VAL_COORDS_PATH
)

print("\nTraining coordinates:", train_coords.shape)
print("Validation coordinates:", val_coords.shape)


# ============================================================
# CREATE DISTANCE-AWARE TARGET
# ============================================================

# The center pixel is 1.0.
#
# Distance:
#
# 0 -> 1.00
# 1 -> 0.75
# 2 -> 0.50
# 3 -> 0.25
#
# This approximately matches the competition's radius-3
# spatial tolerance.

target_kernel = np.zeros(
    (PATCH_SIZE, PATCH_SIZE),
    dtype=np.float32
)

for dy in range(
    -RADIUS,
    RADIUS + 1
):

    for dx in range(
        -RADIUS,
        RADIUS + 1
    ):

        distance = np.sqrt(
            dy * dy + dx * dx
        )

        if distance <= 3.0:

            value = max(
                1.0 - distance / 4.0,
                0.0
            )

            target_kernel[
                dy + RADIUS,
                dx + RADIUS
            ] = value


# ============================================================
# DATASET
# ============================================================

class FaultDataset(Dataset):

    def __init__(
        self,
        coords,
        features,
        labels
    ):

        self.coords = coords
        self.features = features
        self.labels = labels

        self.h = features.shape[1]
        self.w = features.shape[2]

    def __len__(self):

        return len(self.coords)

    def __getitem__(
        self,
        idx
    ):

        y, x = self.coords[idx]

        y = int(y)
        x = int(x)

        r = RADIUS

        patch = self.features[
            :,
            y-r:y+r+1,
            x-r:x+r+1
        ]

        center_label = (
            self.labels[y, x] > 0
        )

        if center_label:

            target = target_kernel.copy()

        else:

            target = np.zeros(
                (PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32
            )

        return (
            torch.from_numpy(
                patch.copy()
            ),
            torch.from_numpy(
                target.copy()
            )
        )


# ============================================================
# BALANCED TRAINING SET
# ============================================================

train_positive = train_coords[
    labels[
        train_coords[:, 0],
        train_coords[:, 1]
    ] > 0
]

train_negative = train_coords[
    labels[
        train_coords[:, 0],
        train_coords[:, 1]
    ] == 0
]

print("\nPositive:", len(train_positive))
print("Negative:", len(train_negative))


# Balance negatives to positives.

rng = np.random.default_rng(SEED)

if len(train_negative) > len(train_positive):

    negative_idx = rng.choice(
        len(train_negative),
        size=len(train_positive),
        replace=False
    )

    train_negative = train_negative[
        negative_idx
    ]


train_coords_balanced = np.concatenate(
    [
        train_positive,
        train_negative
    ],
    axis=0
)

rng.shuffle(
    train_coords_balanced
)

print(
    "Balanced training:",
    train_coords_balanced.shape
)


# ============================================================
# DATASET / LOADER
# ============================================================

train_dataset = FaultDataset(
    train_coords_balanced,
    features,
    labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# U-NET
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class FaultUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = ConvBlock(
            19,
            32
        )

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = ConvBlock(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
        self.pool(e1)
    )

        e3 = self.enc3(
        self.pool(e2)
    )

        b = self.bottleneck(
        self.pool(e3)
    )

        d3 = self.up3(b)

    # Match decoder size to skip connection
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = nn.functional.interpolate(
            d3,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = torch.cat(
        [d3, e3],
        dim=1
    )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = nn.functional.interpolate(
            d2,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = torch.cat(
        [d2, e2],
        dim=1
    )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
           d1 = nn.functional.interpolate(
            d1,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = torch.cat(
        [d1, e1],
        dim=1
    )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# MODEL
# ============================================================

model = FaultUNet().to(
    DEVICE
)

print("\nModel created.")


# ============================================================
# LOSS
# ============================================================

bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [POS_WEIGHT],
        device=DEVICE
    )
)


def distance_tversky(
    logits,
    targets,
    alpha=0.3,
    beta=0.7
):

    probs = torch.sigmoid(
        logits
    )

    tp = (
        probs * targets
    ).sum()

    fp = (
        probs * (1.0 - targets)
    ).sum()

    fn = (
        (1.0 - probs) * targets
    ).sum()

    return (
        tp
        /
        (
            tp
            + alpha * fp
            + beta * fn
            + 1e-7
        )
    )


def loss_fn(
    logits,
    targets
):

    loss_bce = bce(
        logits,
        targets
    )

    tv = distance_tversky(
        logits,
        targets
    )

    loss_tv = 1.0 - tv

    return (
        0.55 * loss_bce
        +
        0.45 * loss_tv
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAIN
# ============================================================

best_loss = float("inf")

print("\nStarting training...\n")


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for batch_idx, (
        x,
        y
    ) in enumerate(train_loader):

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        y = y.unsqueeze(1)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = loss_fn(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

    epoch_loss = (
        running_loss
        /
        len(train_loader)
    )

    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"| Loss: {epoch_loss:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "loss":
                    epoch_loss
            },
            BEST_MODEL_PATH
        )

        print(
            "  ✓ Saved best model"
        )


print("\n" + "=" * 70)
print("V7 TRAINING COMPLETE")
print("=" * 70)

print(
    "\nBest training loss:",
    best_loss
)

print(
    "Saved:",
    BEST_MODEL_PATH
)