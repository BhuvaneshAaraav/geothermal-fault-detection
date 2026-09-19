import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# V6 CONFIG
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

TRAIN_COORDS = "data/processed/unet/train_coords.npy"
VAL_COORDS = "data/processed/unet/val_coords.npy"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "models"

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "fault_v6_unet_best.pt"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "fault_v6_unet.pt"
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 31
RADIUS = 15

BATCH_SIZE = 16

# Screening first
EPOCHS = 2

LR = 3e-4
WEIGHT_DECAY = 1e-4

N_POSITIVE = 44000
N_NEGATIVE = 44000

# Loss weights
BCE_WEIGHT = 0.55
TVERSKY_WEIGHT = 0.45

# Tversky
ALPHA = 0.20
BETA = 0.80

# Positive weighting is deliberately modest.
POS_WEIGHT = 1.5

NUM_WORKERS = 0

SEED = 42


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("Using device:", DEVICE)


# ============================================================
# MODEL
# V1-STYLE U-NET
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
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class FaultSegmentationUNet(nn.Module):

    def __init__(
        self,
        in_channels=19
    ):

        super().__init__()

        # Encoder

        self.enc1 = ConvBlock(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # Decoder

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool3(e3)
        )

        # Decoder 3

        d3 = self.up3(b)

        if d3.shape[-2:] != e3.shape[-2:]:

            d3 = F.interpolate(
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

        # Decoder 2

        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:

            d2 = F.interpolate(
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

        # Decoder 1

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:

            d1 = F.interpolate(
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
# CENTER-PIXEL DATASET
# ============================================================

class CenterDataset(Dataset):

    def __init__(
        self,
        features,
        labels,
        coords,
        means,
        stds
    ):

        self.features = features
        self.labels = labels
        self.coords = coords

        self.means = means
        self.stds = stds

    def __len__(self):

        return len(self.coords)

    def __getitem__(self, idx):

        r, c = self.coords[idx]

        r = int(r)
        c = int(c)

        r0 = r - RADIUS
        r1 = r + RADIUS + 1

        c0 = c - RADIUS
        c1 = c + RADIUS + 1

        patch = self.features[
            :,
            r0:r1,
            c0:c1
        ]

        center_label = self.labels[
            r,
            c
        ]

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        expected_shape = (
            19,
            PATCH_SIZE,
            PATCH_SIZE
        )

        if patch.shape != expected_shape:

            raise RuntimeError(
                f"Bad patch shape {patch.shape} "
                f"at {(r,c)}"
            )

        patch = patch.astype(
            np.float32,
            copy=True
        )

        # ----------------------------------------------------
        # NODATA HANDLING
        # ----------------------------------------------------

        invalid = (
            ~np.isfinite(patch)
            |
            (np.abs(patch) > 1e30)
        )

        if invalid.any():

            for band in range(19):

                bad = invalid[band]

                if bad.any():

                    patch[
                        band,
                        bad
                    ] = self.means[band]

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        patch = (
            patch
            - self.means[:, None, None]
        ) / (
            self.stds[:, None, None]
            + 1e-8
        )

        patch = np.clip(
            patch,
            -10.0,
            10.0
        )

        # ----------------------------------------------------
        # CENTER TARGET ONLY
        # ----------------------------------------------------

        target = np.float32(
            center_label > 0
        )

        return (
            torch.from_numpy(patch),
            torch.tensor(
                target,
                dtype=torch.float32
            )
        )


# ============================================================
# CENTER TVERSKY LOSS
# ============================================================

def center_tversky_loss(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    # Only center pixel

    center = RADIUS

    probs = probs[
        :,
        0,
        center,
        center
    ]

    targets = targets.view(-1)

    tp = (
        probs * targets
    ).sum()

    fp = (
        probs
        * (1.0 - targets)
    ).sum()

    fn = (
        (1.0 - probs)
        * targets
    ).sum()

    score = (
        tp + 1e-6
    ) / (
        tp
        + ALPHA * fp
        + BETA * fn
        + 1e-6
    )

    return 1.0 - score


# ============================================================
# CENTER BCE
# ============================================================

def center_bce_loss(
    logits,
    targets
):

    center = RADIUS

    center_logits = logits[
        :,
        0,
        center,
        center
    ]

    targets = targets.view(-1)

    pos_weight = torch.tensor(
        [POS_WEIGHT],
        device=logits.device
    )

    return F.binary_cross_entropy_with_logits(
        center_logits,
        targets,
        pos_weight=pos_weight
    )


# ============================================================
# V6 LOSS
# ============================================================

def v6_loss(
    logits,
    targets
):

    bce = center_bce_loss(
        logits,
        targets
    )

    tv = center_tversky_loss(
        logits,
        targets
    )

    loss = (
        BCE_WEIGHT * bce
        +
        TVERSKY_WEIGHT * tv
    )

    return loss, bce, tv


# ============================================================
# LOAD RASTER
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print(
    "Feature raster:",
    FEATURE_PATH
)

print(
    "Label raster:",
    LABEL_PATH
)


with rasterio.open(
    FEATURE_PATH
) as src:

    features = src.read()


with rasterio.open(
    LABEL_PATH
) as src:

    labels = src.read(1)


print(
    "Features:",
    features.shape
)

print(
    "Labels:",
    labels.shape
)


# ============================================================
# DATA CHECK
# ============================================================

print()
print("=" * 70)
print("DATA CHECK")
print("=" * 70)

print(
    "Bands:",
    features.shape[0]
)

print(
    "Height:",
    features.shape[1]
)

print(
    "Width:",
    features.shape[2]
)

print(
    "Fault pixels:",
    int(np.sum(labels > 0))
)


# ============================================================
# NORMALIZATION
# ============================================================

means = np.load(
    MEANS_PATH
).astype(
    np.float32
)

stds = np.load(
    STDS_PATH
).astype(
    np.float32
)

print()
print("=" * 70)
print("NORMALIZATION")
print("=" * 70)

print(
    "Means:",
    means.shape
)

print(
    "Stds:",
    stds.shape
)


# ============================================================
# COORDINATES
# ============================================================

train_coords = np.load(
    TRAIN_COORDS
)

val_coords = np.load(
    VAL_COORDS
)

print()
print("=" * 70)
print("COORDINATES")
print("=" * 70)

print(
    "Training:",
    train_coords.shape
)

print(
    "Validation:",
    val_coords.shape
)


# ============================================================
# TRAIN POSITIVE / NEGATIVE
# ============================================================

train_labels = labels[
    train_coords[:, 0],
    train_coords[:, 1]
]

positive_mask = (
    train_labels > 0
)

positive_coords = (
    train_coords[positive_mask]
)

negative_coords = (
    train_coords[~positive_mask]
)

print()
print("=" * 70)
print("TRAINING DATA")
print("=" * 70)

print(
    "Positive coordinates:",
    len(positive_coords)
)

print(
    "Negative coordinates:",
    len(negative_coords)
)


# ============================================================
# RANDOM SAMPLING
# ============================================================

def random_sample(
    coords,
    n
):

    if len(coords) <= n:

        return coords.copy()

    indices = np.random.choice(
        len(coords),
        size=n,
        replace=False
    )

    return coords[
        indices
    ]


positive_sample = random_sample(
    positive_coords,
    N_POSITIVE
)

negative_sample = random_sample(
    negative_coords,
    N_NEGATIVE
)


# ============================================================
# COMBINE
# ============================================================

train_all = np.concatenate(
    [
        positive_sample,
        negative_sample
    ],
    axis=0
)


shuffle_idx = np.random.permutation(
    len(train_all)
)

train_all = train_all[
    shuffle_idx
]


print()
print("=" * 70)
print("V6 TRAINING SET")
print("=" * 70)

print(
    "Positive:",
    len(positive_sample)
)

print(
    "Negative:",
    len(negative_sample)
)

print(
    "Total:",
    len(train_all)
)


# ============================================================
# VALIDATION SCREENING SET
# ============================================================

VAL_SCREEN_SIZE = min(
    20000,
    len(val_coords)
)

val_indices = np.random.choice(
    len(val_coords),
    size=VAL_SCREEN_SIZE,
    replace=False
)

val_screen_coords = (
    val_coords[val_indices]
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = CenterDataset(
    features,
    labels,
    train_all,
    means,
    stds
)

val_dataset = CenterDataset(
    features,
    labels,
    val_screen_coords,
    means,
    stds
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


print()
print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING V6 MODEL")
print("=" * 70)

model = FaultSegmentationUNet(
    in_channels=19
).to(DEVICE)

print(model)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    all_probs = []
    all_targets = []

    for x, y in val_loader:

        x = x.to(
            DEVICE
        )

        logits = model(x)

        probs = torch.sigmoid(
            logits
        )

        center_probs = probs[
            :,
            0,
            RADIUS,
            RADIUS
        ]

        all_probs.append(
            center_probs.cpu().numpy()
        )

        all_targets.append(
            y.numpy()
        )

    probs = np.concatenate(
        all_probs
    )

    targets = np.concatenate(
        all_targets
    )

    roc = roc_auc_score(
        targets,
        probs
    )

    pr = average_precision_score(
        targets,
        probs
    )

    return (
        roc,
        pr,
        probs,
        targets
    )


# ============================================================
# TRAINING
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_pr = -1.0

print()
print("=" * 70)
print("STARTING V6 SCREENING")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_tv = 0.0

    for batch_idx, (
        x,
        y
    ) in enumerate(
        train_loader
    ):

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad()

        logits = model(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss, bce, tv = v6_loss(
            logits,
            y
        )

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_bce += bce.item()

        total_tv += tv.item()

        if (
            batch_idx + 1
        ) % 500 == 0:

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    # ========================================================
    # AVERAGE
    # ========================================================

    num_batches = len(
        train_loader
    )

    avg_loss = (
        total_loss
        / num_batches
    )

    avg_bce = (
        total_bce
        / num_batches
    )

    avg_tv = (
        total_tv
        / num_batches
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    roc, pr, probs, targets = (
        evaluate()
    )

    # ========================================================
    # PROBABILITY STATISTICS
    # ========================================================

    p50 = np.percentile(
        probs,
        50
    )

    p90 = np.percentile(
        probs,
        90
    )

    p99 = np.percentile(
        probs,
        99
    )

    count_05 = np.sum(
        probs >= 0.5
    )

    count_08 = np.sum(
        probs >= 0.8
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss   : {avg_loss:.6f}"
    )

    print(
        f"Train BCE    : {avg_bce:.6f}"
    )

    print(
        f"Train Tversky: {avg_tv:.6f}"
    )

    print(
        f"Val ROC-AUC  : {roc:.6f}"
    )

    print(
        f"Val PR-AUC   : {pr:.6f}"
    )

    print()
    print(
        "Probability diagnostics:"
    )

    print(
        f"Median       : {p50:.6f}"
    )

    print(
        f"90th percentile: {p90:.6f}"
    )

    print(
        f"99th percentile: {p99:.6f}"
    )

    print(
        ">= 0.5:",
        int(count_05)
    )

    print(
        ">= 0.8:",
        int(count_08)
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if pr > best_pr:

        best_pr = pr

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_pr_auc":
                    pr,

                "val_roc_auc":
                    roc
            },
            BEST_MODEL
        )

        print()
        print(
            "✓ BEST MODEL SAVED"
        )

        print(
            f"Best PR-AUC: {best_pr:.6f}"
        )

    print("=" * 70)


# ============================================================
# SAVE FINAL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "epoch":
            EPOCHS,

        "val_pr_auc":
            best_pr
    },
    FINAL_MODEL
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("V6 SCREENING COMPLETE")
print("=" * 70)

print(
    f"Best PR-AUC: {best_pr:.6f}"
)

print(
    "Best model:",
    BEST_MODEL
)

print(
    "Final model:",
    FINAL_MODEL
)

print()
print(
    "STOP HERE."
)

print(
    "Do not increase V6 to 8 epochs yet."
)

print(
    "We first inspect the 2-epoch result."
)