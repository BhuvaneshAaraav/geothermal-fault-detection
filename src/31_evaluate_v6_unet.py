import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio

from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

MODEL_PATH = "models/fault_v6_unet_best.pt"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

TRAIN_COORDS = "data/processed/unet/train_coords.npy"
VAL_COORDS = "data/processed/unet/val_coords.npy"

OUTPUT_PROB = "outputs/v6_validation_probability.npy"

PATCH_SIZE = 31
RADIUS = 15

BATCH_SIZE = 16


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
# EXACT V1/V6 ARCHITECTURE
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
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

        self.enc1 = ConvBlock(
            in_channels, 32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(
            32, 64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            64, 128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            128, 256
        )

        self.up3 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )

        self.dec3 = ConvBlock(
            256, 128
        )

        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec2 = ConvBlock(
            128, 64
        )

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.dec1 = ConvBlock(
            64, 32
        )

        self.final = nn.Conv2d(
            32, 1, 1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

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
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

with rasterio.open(FEATURE_PATH) as src:
    features = src.read()

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

means = np.load(
    MEANS_PATH
).astype(np.float32)

stds = np.load(
    STDS_PATH
).astype(np.float32)

val_coords = np.load(
    VAL_COORDS
)

print("Features:", features.shape)
print("Labels:", labels.shape)
print("Validation coordinates:", val_coords.shape)


# ============================================================
# IMPORTANT:
# USE ONLY VALID INTERIOR VALIDATION CENTERS
# ============================================================

height = features.shape[1]
width = features.shape[2]

safe_mask = (
    (val_coords[:, 0] >= RADIUS)
    &
    (val_coords[:, 0] < height - RADIUS)
    &
    (val_coords[:, 1] >= RADIUS)
    &
    (val_coords[:, 1] < width - RADIUS)
)

val_coords = val_coords[safe_mask]

print(
    "Safe validation coordinates:",
    val_coords.shape
)


# ============================================================
# VALIDATION MASK
# ============================================================

# The validation coordinates already represent the spatial
# validation blocks. We construct a mask from them.

validation_mask = np.zeros(
    (height, width),
    dtype=bool
)

validation_mask[
    val_coords[:, 0],
    val_coords[:, 1]
] = True

print(
    "Validation pixels:",
    validation_mask.sum()
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING V6 MODEL")
print("=" * 70)

model = FaultSegmentationUNet(
    in_channels=19
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print(
    "Loaded:",
    MODEL_PATH
)

if "epoch" in checkpoint:

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

if "val_pr_auc" in checkpoint:

    print(
        "Checkpoint PR-AUC:",
        checkpoint["val_pr_auc"]
    )


# ============================================================
# FULL RASTER PREDICTION
# ============================================================

print()
print("=" * 70)
print("FULL RASTER INFERENCE")
print("=" * 70)

probability_map = np.zeros(
    (height, width),
    dtype=np.float32
)

# Count predictions
total = len(val_coords)

processed = 0

with torch.no_grad():

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_coords = val_coords[
            start:end
        ]

        patches = []

        for r, c in batch_coords:

            r = int(r)
            c = int(c)

            r0 = r - RADIUS
            r1 = r + RADIUS + 1

            c0 = c - RADIUS
            c1 = c + RADIUS + 1

            patch = features[
                :,
                r0:r1,
                c0:c1
            ].astype(
                np.float32,
                copy=True
            )

            # ------------------------------------------------
            # NODATA
            # ------------------------------------------------

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
                        ] = means[band]

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            patch = (
                patch
                - means[:, None, None]
            ) / (
                stds[:, None, None]
                + 1e-8
            )

            patch = np.clip(
                patch,
                -10.0,
                10.0
            )

            patches.append(
                patch
            )

        x = np.stack(
            patches,
            axis=0
        )

        x = torch.from_numpy(
            x
        ).to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(
            logits
        )

        # Center prediction
        center_probs = probs[
            :,
            0,
            RADIUS,
            RADIUS
        ]

        center_probs = (
            center_probs
            .cpu()
            .numpy()
        )

        rows = batch_coords[:, 0]
        cols = batch_coords[:, 1]

        probability_map[
            rows,
            cols
        ] = center_probs

        processed = end

        if processed % 50000 < BATCH_SIZE:

            print(
                f"Processed "
                f"{processed:,}/{total:,}"
            )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

np.save(
    OUTPUT_PROB,
    probability_map
)

print()
print(
    "Saved:",
    OUTPUT_PROB
)


# ============================================================
# BASIC STATISTICS
# ============================================================

eval_probs = probability_map[
    validation_mask
]

eval_labels = labels[
    validation_mask
]

print()
print("=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)

print(
    "Evaluation pixels:",
    len(eval_probs)
)

print(
    "Fault pixels:",
    int(np.sum(eval_labels > 0))
)

print(
    "Min:",
    float(eval_probs.min())
)

print(
    "Max:",
    float(eval_probs.max())
)

print(
    "Mean:",
    float(eval_probs.mean())
)

print(
    "Median:",
    float(np.median(eval_probs))
)

print(
    "P90:",
    float(np.percentile(
        eval_probs,
        90
    ))
)

print(
    "P99:",
    float(np.percentile(
        eval_probs,
        99
    ))
)


# ============================================================
# THRESHOLD COUNTS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD COUNTS")
print("=" * 70)

for threshold in [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]:

    count = np.sum(
        eval_probs >= threshold
    )

    print(
        f">= {threshold:.1f}:",
        int(count)
    )


# ============================================================
# ROC / PR
# ============================================================

roc = roc_auc_score(
    eval_labels,
    eval_probs
)

pr = average_precision_score(
    eval_labels,
    eval_probs
)

print()
print("=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

print(
    f"ROC-AUC: {roc:.6f}"
)

print(
    f"PR-AUC : {pr:.6f}"
)


# ============================================================
# THRESHOLD PRECISION / RECALL
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

for threshold in [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]:

    pred = (
        eval_probs >= threshold
    )

    tp = np.sum(
        pred & (eval_labels > 0)
    )

    fp = np.sum(
        pred & (eval_labels == 0)
    )

    fn = np.sum(
        (~pred) & (eval_labels > 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    print(
        f"Threshold {threshold:.1f} | "
        f"Pred={int(pred.sum()):,} | "
        f"TP={int(tp):,} | "
        f"FP={int(fp):,} | "
        f"FN={int(fn):,} | "
        f"Precision={precision:.6f} | "
        f"Recall={recall:.6f}"
    )


print()
print("=" * 70)
print("V6 FULL-RASTER EVALUATION COMPLETE")
print("=" * 70)