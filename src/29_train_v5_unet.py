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
# V5 CONFIGURATION
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

TRAIN_COORDS = "data/processed/unet/train_coords.npy"
VAL_COORDS = "data/processed/unet/val_coords.npy"

HARD_COORDS = "data/processed/hard_negatives/hard_negative_coords.npy"
HARD_SCORES = "data/processed/hard_negatives/hard_negative_scores.npy"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "models"

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "fault_v5_unet_best.pt"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "fault_v5_unet.pt"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

BATCH_SIZE = 16

# IMPORTANT:
# First run only 2 epochs as a screening experiment.
EPOCHS = 2

LR = 3e-4
WEIGHT_DECAY = 1e-4

# Training composition
N_POSITIVE = 44000
N_ORDINARY_NEGATIVE = 44000
N_HARD_NEGATIVE = 12000

# Tversky
ALPHA = 0.20
BETA = 0.80

# Focal
FOCAL_GAMMA = 2.0

NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
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
# SAME BASIC U-NET ARCHITECTURE AS V1
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class FaultSegmentationUNet(nn.Module):

    def __init__(self, in_channels=19):

        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

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

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # -------------------------
        # Decoder
        # -------------------------

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

        # -------------------------
        # Output
        # -------------------------

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
# DATASET
# ============================================================

class PatchDataset(Dataset):

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

        # ----------------------------------------------------
        # Feature patch
        # ----------------------------------------------------

        patch = self.features[
            :,
            r0:r1,
            c0:c1
        ]

        # ----------------------------------------------------
        # Label patch
        # ----------------------------------------------------

        target = self.labels[
            r0:r1,
            c0:c1
        ]

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        expected_feature_shape = (
            19,
            PATCH_SIZE,
            PATCH_SIZE
        )

        expected_target_shape = (
            PATCH_SIZE,
            PATCH_SIZE
        )

        if patch.shape != expected_feature_shape:

            raise RuntimeError(
                f"Invalid feature patch shape "
                f"{patch.shape} at {(r, c)}"
            )

        if target.shape != expected_target_shape:

            raise RuntimeError(
                f"Invalid target patch shape "
                f"{target.shape} at {(r, c)}"
            )

        # ----------------------------------------------------
        # Float32
        # ----------------------------------------------------

        patch = patch.astype(
            np.float32,
            copy=True
        )

        # ----------------------------------------------------
        # Detect raster nodata
        #
        # The TIFF contains values around
        # -3.40282e+38.
        # ----------------------------------------------------

        invalid = (
            ~np.isfinite(patch)
            |
            (np.abs(patch) > 1e30)
        )

        # Replace invalid values with band mean

        if invalid.any():

            for band in range(19):

                bad = invalid[band]

                if bad.any():

                    patch[
                        band,
                        bad
                    ] = self.means[band]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        patch = (
            patch
            - self.means[:, None, None]
        ) / (
            self.stds[:, None, None]
            + 1e-8
        )

        # ----------------------------------------------------
        # Protect against extreme values
        # ----------------------------------------------------

        patch = np.clip(
            patch,
            -10.0,
            10.0
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = (
            target > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        x = torch.from_numpy(
            patch
        )

        y = torch.from_numpy(
            target
        ).unsqueeze(0)

        return x, y


# ============================================================
# TVERSKY LOSS
# ============================================================

def tversky_loss(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    probs = probs.reshape(
        probs.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    tp = (
        probs * targets
    ).sum(dim=1)

    fp = (
        probs
        * (1.0 - targets)
    ).sum(dim=1)

    fn = (
        (1.0 - probs)
        * targets
    ).sum(dim=1)

    score = (
        tp + 1e-6
    ) / (
        tp
        + ALPHA * fp
        + BETA * fn
        + 1e-6
    )

    return 1.0 - score.mean()


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    pt = (
        probs * targets
        +
        (1.0 - probs)
        * (1.0 - targets)
    )

    loss = (
        (1.0 - pt)
        ** FOCAL_GAMMA
    ) * bce

    return loss.mean()


# ============================================================
# FALSE POSITIVE SUPPRESSION
# ============================================================

def false_positive_loss(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )

    background = (
        1.0 - targets
    )

    false_positive_probability = (
        probs * background
    )

    # Strongly penalize confident background
    # predictions.

    loss = (
        false_positive_probability
        ** 2
    ).mean()

    return loss


# ============================================================
# V5 COMBINED LOSS
# ============================================================

def v5_loss(
    logits,
    targets
):

    tv = tversky_loss(
        logits,
        targets
    )

    focal = focal_loss(
        logits,
        targets
    )

    fp = false_positive_loss(
        logits,
        targets
    )

    loss = (
        0.55 * tv
        +
        0.30 * focal
        +
        0.15 * fp
    )

    return (
        loss,
        tv,
        focal,
        fp
    )


# ============================================================
# LOAD RASTER DATA
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
    "Label raster  :",
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
    "Labels  :",
    labels.shape
)


# ============================================================
# CHECK DATA
# ============================================================

print()
print("=" * 70)
print("DATA CHECK")
print("=" * 70)

print(
    "Number of bands:",
    features.shape[0]
)

print(
    "Raster height:",
    features.shape[1]
)

print(
    "Raster width:",
    features.shape[2]
)

print(
    "Total fault pixels:",
    np.sum(labels > 0)
)


# ============================================================
# LOAD NORMALIZATION
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
    "Means shape:",
    means.shape
)

print(
    "Stds shape:",
    stds.shape
)


# ============================================================
# LOAD COORDINATES
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
# IDENTIFY POSITIVE / NEGATIVE TRAINING CENTERS
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

ordinary_negative_coords = (
    train_coords[~positive_mask]
)

print()
print("=" * 70)
print("TRAINING COORDINATES")
print("=" * 70)

print(
    "Positive:",
    len(positive_coords)
)

print(
    "Ordinary negative:",
    len(ordinary_negative_coords)
)


# ============================================================
# LOAD HARD NEGATIVES
# ============================================================

hard_coords = np.load(
    HARD_COORDS
)

hard_scores = np.load(
    HARD_SCORES
)

print()
print("=" * 70)
print("HARD NEGATIVES")
print("=" * 70)

print(
    "Hard pool:",
    hard_coords.shape
)

print(
    "Score range:",
    hard_scores.min(),
    "to",
    hard_scores.max()
)


# ============================================================
# CONTROLLED HARD NEGATIVE SELECTION
# ============================================================

# We intentionally use only moderately hard examples.
#
# V2 used too much hard-negative pressure and performance
# decreased.
#
# We therefore avoid the most extreme 0.90+ predictions.

moderate_mask = (
    (hard_scores >= 0.70)
    &
    (hard_scores < 0.90)
)

moderate_hard_coords = (
    hard_coords[moderate_mask]
)

moderate_hard_scores = (
    hard_scores[moderate_mask]
)

print(
    "Moderate hard negatives:",
    moderate_hard_coords.shape
)


# ============================================================
# REMOVE ACTUAL FAULT PIXELS
# ============================================================

verified_hard = []

for r, c in moderate_hard_coords:

    r = int(r)
    c = int(c)

    if labels[r, c] == 0:

        verified_hard.append(
            [r, c]
        )


verified_hard = np.asarray(
    verified_hard,
    dtype=np.int32
)


print(
    "Verified hard negatives:",
    verified_hard.shape
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

    return coords[indices]


positive_sample = random_sample(
    positive_coords,
    N_POSITIVE
)

ordinary_negative_sample = random_sample(
    ordinary_negative_coords,
    N_ORDINARY_NEGATIVE
)

hard_negative_sample = random_sample(
    verified_hard,
    N_HARD_NEGATIVE
)


# ============================================================
# COMBINE TRAINING COORDINATES
# ============================================================

train_all = np.concatenate(
    [
        positive_sample,
        ordinary_negative_sample,
        hard_negative_sample
    ],
    axis=0
)


# Shuffle

shuffle_indices = np.random.permutation(
    len(train_all)
)

train_all = train_all[
    shuffle_indices
]


print()
print("=" * 70)
print("V5 TRAINING DATA")
print("=" * 70)

print(
    "Positive samples :",
    len(positive_sample)
)

print(
    "Ordinary negative:",
    len(ordinary_negative_sample)
)

print(
    "Hard negatives   :",
    len(hard_negative_sample)
)

print(
    "TOTAL            :",
    len(train_all)
)


# ============================================================
# DATASET
# ============================================================

train_dataset = PatchDataset(
    features,
    labels,
    train_all,
    means,
    stds
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


val_dataset = PatchDataset(
    features,
    labels,
    val_screen_coords,
    means,
    stds
)


print()
print(
    "Validation screening samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING V5 MODEL")
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

        x = x.to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(
            logits
        )

        # Center pixel only for quick screening

        center = RADIUS

        center_probs = probs[
            :,
            0,
            center,
            center
        ]

        center_targets = y[
            :,
            0,
            center,
            center
        ]

        all_probs.append(
            center_probs.cpu().numpy()
        )

        all_targets.append(
            center_targets.numpy()
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

    return roc, pr, probs, targets


# ============================================================
# TRAIN
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_pr = -1.0

print()
print("=" * 70)
print("STARTING V5 SCREENING")
print("=" * 70)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LR
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    total_tv = 0.0
    total_focal = 0.0
    total_fp = 0.0

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

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad()

        logits = model(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss, tv, focal, fp = v5_loss(
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

        total_tv += tv.item()

        total_focal += focal.item()

        total_fp += fp.item()

        if (
            batch_idx + 1
        ) % 500 == 0:

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    # ========================================================
    # EPOCH STATISTICS
    # ========================================================

    num_batches = len(
        train_loader
    )

    train_loss = (
        total_loss / num_batches
    )

    train_tv = (
        total_tv / num_batches
    )

    train_focal = (
        total_focal / num_batches
    )

    train_fp = (
        total_fp / num_batches
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    roc, pr, val_probs, val_targets = (
        evaluate()
    )

    # ========================================================
    # PROBABILITY DIAGNOSTICS
    # ========================================================

    p50 = np.percentile(
        val_probs,
        50
    )

    p90 = np.percentile(
        val_probs,
        90
    )

    p99 = np.percentile(
        val_probs,
        99
    )

    count_05 = np.sum(
        val_probs >= 0.5
    )

    count_08 = np.sum(
        val_probs >= 0.8
    )

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss    : {train_loss:.6f}"
    )

    print(
        f"Train Tversky : {train_tv:.6f}"
    )

    print(
        f"Train Focal   : {train_focal:.6f}"
    )

    print(
        f"Train FP Loss : {train_fp:.6f}"
    )

    print(
        f"Val ROC-AUC   : {roc:.6f}"
    )

    print(
        f"Val PR-AUC    : {pr:.6f}"
    )

    print()
    print(
        "Probability diagnostics:"
    )

    print(
        f"Median        : {p50:.6f}"
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
    # SAVE BEST
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
            "Best PR-AUC:",
            best_pr
        )

    print("=" * 70)


# ============================================================
# SAVE FINAL MODEL
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
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("V5 SCREENING COMPLETE")
print("=" * 70)

print(
    "Best validation PR-AUC:",
    best_pr
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
    "IMPORTANT:"
)

print(
    "This was only the 2-epoch screening run."
)

print(
    "Do NOT run 8 epochs yet."
)

print(
    "Next step: evaluate the V5 best checkpoint"
)

print(
    "on the FULL spatial validation raster"
)

print(
    "using the VERIFIED competition DTI metric."
)