import os
import random
import numpy as np
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

FEATURE_RASTER = "data/raw/training_features.tif"
LABEL_RASTER = "data/raw/Training_fault_labels.tif"

TRAIN_COORDS = "data/processed/unet/train_coords.npy"
TRAIN_LABELS = "data/processed/unet/train_labels.npy"

VAL_COORDS = "data/processed/unet/val_coords.npy"
VAL_LABELS = "data/processed/unet/val_labels.npy"

MEANS_FILE = "data/processed/cnn/feature_means.npy"
STDS_FILE = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "models"
BEST_MODEL = os.path.join(MODEL_DIR, "fault_tversky_unet_best.pt")
FINAL_MODEL = os.path.join(MODEL_DIR, "fault_tversky_unet.pt")

SEED = 42

PATCH_SIZE = 31
RADIUS = 3

BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 1e-3

# Loss weighting
BCE_WEIGHT = 0.50
TVERSKY_WEIGHT = 0.50

# Competition metric parameters
ALPHA = 0.2
BETA = 0.8

# Validation subset
VAL_MAX_SAMPLES = 20000

# Gradient clipping
GRAD_CLIP = 2.0


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
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# ============================================================
# DATASET
# ============================================================

class FaultPatchDataset(Dataset):

    def __init__(
        self,
        feature_array,
        label_array,
        valid_array,
        coords,
        patch_size=31
    ):

        self.features = feature_array
        self.labels = label_array
        self.valid = valid_array

        self.coords = coords.astype(np.int32)

        self.patch_size = patch_size
        self.radius = patch_size // 2

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        y, x = self.coords[idx]

        r = self.radius

        feature_patch = self.features[
            :,
            y-r:y+r+1,
            x-r:x+r+1
        ]

        label_patch = self.labels[
            y-r:y+r+1,
            x-r:x+r+1
        ]

        valid_patch = self.valid[
            y-r:y+r+1,
            x-r:x+r+1
        ]

        # Copy because raster slices are usually non-contiguous
        feature_patch = np.asarray(
            feature_patch,
            dtype=np.float32
        ).copy()

        label_patch = np.asarray(
            label_patch,
            dtype=np.float32
        ).copy()

        valid_patch = np.asarray(
            valid_patch,
            dtype=np.float32
        ).copy()

        return (
            torch.from_numpy(feature_patch),
            torch.from_numpy(label_patch).unsqueeze(0),
            torch.from_numpy(valid_patch).unsqueeze(0)
        )


# ============================================================
# U-NET BUILDING BLOCK
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


# ============================================================
# REAL PATCH SEGMENTATION U-NET
# ============================================================

class FaultSegmentationUNet(nn.Module):

    def __init__(self, in_channels=19):

        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)

        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(64, 32)

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

        # Decoder
        d3 = self.up3(b)

        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = nn.functional.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = nn.functional.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# BCE LOSS
# ============================================================

def masked_bce_loss(
    logits,
    targets,
    valid_mask
):

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(
            4.0,
            device=logits.device
        )
    )

    # Only valid pixels
    loss = loss * valid_mask

    denominator = valid_mask.sum().clamp_min(1.0)

    return loss.sum() / denominator


# ============================================================
# ZERO-PADDING SHIFT
#
# output[y,x] = input[y+dy,x+dx]
# ============================================================

def shift_zero(x, dy, dx):

    B, C, H, W = x.shape

    result = torch.zeros_like(x)

    src_y0 = max(0, dy)
    src_y1 = min(H, H + dy)

    dst_y0 = max(0, -dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, dx)
    src_x1 = min(W, W + dx)

    dst_x0 = max(0, -dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if (
        src_y1 > src_y0
        and src_x1 > src_x0
    ):

        result[
            :,
            :,
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = x[
            :,
            :,
            src_y0:src_y1,
            src_x0:src_x1
        ]

    return result


# ============================================================
# DISTANCE-WEIGHTED TVERSKY
#
# This approximates the competition's spatial tolerance:
#
# R = 3 pixels
# kernel(d) = max(1 - d/R, 0)
#
# TP:
# max prediction around each GT pixel
#
# FN:
# remaining GT probability
#
# FP:
# prediction outside GT influence
# ============================================================

def distance_tversky_score(
    probabilities,
    targets,
    valid_mask,
    radius=3,
    alpha=0.2,
    beta=0.8
):

    device = probabilities.device

    # --------------------------------------------------------
    # Only evaluate the inner region.
    #
    # This prevents artificial boundary effects caused by
    # the 31x31 training patch.
    # --------------------------------------------------------

    margin = radius

    if probabilities.shape[-1] <= 2 * margin:
        raise ValueError("Patch too small for selected radius.")

    p = probabilities[
        :,
        :,
        margin:-margin,
        margin:-margin
    ]

    t = targets[
        :,
        :,
        margin:-margin,
        margin:-margin
    ]

    valid = valid_mask[
        :,
        :,
        margin:-margin,
        margin:-margin
    ]

    # --------------------------------------------------------
    # TP matching
    #
    # For every GT pixel:
    #
    # max p(x) * kernel(distance)
    # --------------------------------------------------------

    tp_match = torch.zeros_like(p)

    # --------------------------------------------------------
    # GT influence for FP calculation
    # --------------------------------------------------------

    gt_influence = torch.zeros_like(t)

    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            distance = np.sqrt(
                float(dx * dx + dy * dy)
            )

            if distance > radius:
                continue

            kernel = 1.0 - distance / radius

            # Prediction around GT
            shifted_p = shift_zero(
                probabilities,
                dy,
                dx
            )

            shifted_p = shifted_p[
                :,
                :,
                margin:-margin,
                margin:-margin
            ]

            candidate_tp = (
                shifted_p * kernel
            )

            tp_match = torch.maximum(
                tp_match,
                candidate_tp
            )

            # ------------------------------------------------
            # GT around prediction
            # ------------------------------------------------

            shifted_t = shift_zero(
                targets,
                -dy,
                -dx
            )

            shifted_t = shifted_t[
                :,
                :,
                margin:-margin,
                margin:-margin
            ]

            candidate_gt = (
                shifted_t * kernel
            )

            gt_influence = torch.maximum(
                gt_influence,
                candidate_gt
            )

    # --------------------------------------------------------
    # Apply valid pixels
    # --------------------------------------------------------

    tp_match = tp_match * valid
    gt_influence = gt_influence * valid

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    tp = (
        t * tp_match
    ).sum(dim=(1, 2, 3))

    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    fn = (
        t * (1.0 - tp_match)
    ).sum(dim=(1, 2, 3))

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    fp = (
        p * (1.0 - gt_influence) * valid
    ).sum(dim=(1, 2, 3))

    # --------------------------------------------------------
    # Tversky
    # --------------------------------------------------------

    denominator = (
        tp
        + alpha * fp
        + beta * fn
        + 1e-7
    )

    score = tp / denominator

    # Patches with no GT:
    #
    # they should mainly be handled by BCE.
    #
    # Do not allow empty patches to dominate the Tversky
    # average.
    has_gt = (
        t.sum(dim=(1, 2, 3)) > 0
    )

    return score, has_gt


# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("LOADING COORDINATES")
print("============================================================")

train_coords = np.load(TRAIN_COORDS)
train_center_labels = np.load(TRAIN_LABELS)

val_coords = np.load(VAL_COORDS)
val_center_labels = np.load(VAL_LABELS)

print("Training coordinates:", train_coords.shape)
print("Validation coordinates:", val_coords.shape)

print(
    "Training positive centers:",
    int(train_center_labels.sum())
)

print(
    "Validation positive centers:",
    int(val_center_labels.sum())
)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print()
print("============================================================")
print("LOADING NORMALIZATION")
print("============================================================")

means = np.load(MEANS_FILE).astype(np.float32)
stds = np.load(STDS_FILE).astype(np.float32)

print("Means:", means.shape)
print("Stds :", stds.shape)


# ============================================================
# LOAD COMPLETE FEATURE RASTER INTO MEMORY
#
# 19 x 3730 x 3292 float32
# ≈ 0.93 GB
#
# This removes the huge rasterio-per-patch I/O bottleneck.
# ============================================================

print()
print("============================================================")
print("LOADING FEATURES INTO RAM")
print("============================================================")

with rasterio.open(FEATURE_RASTER) as src:

    features = src.read(
        out_dtype="float32"
    )

    feature_mask = src.read_masks(1) > 0

print("Feature array:", features.shape)

# ------------------------------------------------------------
# Normalize in-place
# ------------------------------------------------------------

# ------------------------------------------------------------
# Remove invalid / non-finite values BEFORE normalization
# ------------------------------------------------------------

for band in range(features.shape[0]):

    band_data = features[band]

    # Replace invalid pixels with the training mean.
    # This prevents nodata values such as -3.4e38 from
    # overflowing during normalization.
    invalid = (
        (~feature_mask)
        | (~np.isfinite(band_data))
    )

    band_data[invalid] = means[band]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    band_data -= means[band]

    band_data /= max(
        stds[band],
        1e-8
    )

    # --------------------------------------------------------
    # Safety clipping
    # --------------------------------------------------------

    np.clip(
        band_data,
        -10.0,
        10.0,
        out=band_data
    )

# ------------------------------------------------------------
# Explicitly create valid mask for the loss
# ------------------------------------------------------------

valid_array = feature_mask.astype(
    np.float32
)

# Invalid features are now represented by zero after
# normalization.
features[:, ~feature_mask] = 0.0

features = np.ascontiguousarray(
    features,
    dtype=np.float32
)

valid_array = feature_mask.astype(
    np.float32
)

del feature_mask

print("Normalized features ready.")


# ============================================================
# LOAD LABEL RASTER
# ============================================================

print()
print("============================================================")
print("LOADING LABELS")
print("============================================================")

with rasterio.open(LABEL_RASTER) as src:

    labels = src.read(
        1,
        out_dtype="uint8"
    )

labels = labels.astype(
    np.float32
)

print("Labels:", labels.shape)

print(
    "Total fault pixels:",
    int(labels.sum())
)


# ============================================================
# VALIDATION SUBSET
# ============================================================

if len(val_coords) > VAL_MAX_SAMPLES:

    rng = np.random.default_rng(SEED)

    selected = rng.choice(
        len(val_coords),
        size=VAL_MAX_SAMPLES,
        replace=False
    )

    val_coords_eval = val_coords[selected]

    val_center_labels_eval = (
        val_center_labels[selected]
    )

else:

    val_coords_eval = val_coords
    val_center_labels_eval = val_center_labels


print()
print(
    "Validation samples used:",
    len(val_coords_eval)
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = FaultPatchDataset(
    features,
    labels,
    valid_array,
    train_coords,
    PATCH_SIZE
)

val_dataset = FaultPatchDataset(
    features,
    labels,
    valid_array,
    val_coords_eval,
    PATCH_SIZE
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


print()
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# MODEL
# ============================================================

print()
print("============================================================")
print("CREATING SEGMENTATION U-NET")
print("============================================================")

model = FaultSegmentationUNet(
    in_channels=19
).to(DEVICE)

print(model)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-6
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    running_bce = 0.0
    running_tv = 0.0

    count = 0

    for batch_idx, (
        x,
        target,
        valid
    ) in enumerate(train_loader):

        x = x.to(
            DEVICE,
            dtype=torch.float32
        )

        target = target.to(
            DEVICE,
            dtype=torch.float32
        )

        valid = valid.to(
            DEVICE,
            dtype=torch.float32
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        # ----------------------------------------------------
        # BCE
        # ----------------------------------------------------

        bce = masked_bce_loss(
            logits,
            target,
            valid
        )

        # ----------------------------------------------------
        # Distance-aware Tversky
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        tv_score, has_gt = distance_tversky_score(
            probabilities,
            target,
            valid,
            radius=RADIUS,
            alpha=ALPHA,
            beta=BETA
        )

        if has_gt.any():

            tv_loss = (
                1.0
                - tv_score[has_gt].mean()
            )

        else:

            tv_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        loss = (
            BCE_WEIGHT * bce
            + TVERSKY_WEIGHT * tv_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_size = x.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        running_bce += (
            bce.item() * batch_size
        )

        running_tv += (
            tv_loss.item() * batch_size
        )

        count += batch_size

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 200 == 0
            or batch_idx == len(train_loader) - 1
        ):

            print(
                f"\rBatch "
                f"{batch_idx + 1}/{len(train_loader)} "
                f"| Loss {loss.item():.4f}",
                end=""
            )

    print()

    return (
        running_loss / count,
        running_bce / count,
        running_tv / count
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    total_bce = 0.0

    all_center_probs = []
    all_center_labels = []

    all_tv_scores = []

    count = 0

    center_index = PATCH_SIZE // 2

    for batch_idx, (
        x,
        target,
        valid
    ) in enumerate(val_loader):

        x = x.to(
            DEVICE,
            dtype=torch.float32
        )

        target = target.to(
            DEVICE,
            dtype=torch.float32
        )

        valid = valid.to(
            DEVICE,
            dtype=torch.float32
        )

        logits = model(x)

        bce = masked_bce_loss(
            logits,
            target,
            valid
        )

        probabilities = torch.sigmoid(
            logits
        )

        tv_score, has_gt = distance_tversky_score(
            probabilities,
            target,
            valid,
            radius=RADIUS,
            alpha=ALPHA,
            beta=BETA
        )

        if has_gt.any():

            tv_loss = (
                1.0
                - tv_score[has_gt].mean()
            )

            all_tv_scores.extend(
                tv_score[has_gt]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

        else:

            tv_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

        loss = (
            BCE_WEIGHT * bce
            + TVERSKY_WEIGHT * tv_loss
        )

        batch_size = x.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_bce += (
            bce.item() * batch_size
        )

        count += batch_size

        # ----------------------------------------------------
        # Center prediction
        # ----------------------------------------------------

        center_prob = probabilities[
            :,
            0,
            center_index,
            center_index
        ]

        all_center_probs.extend(
            center_prob
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        # Actual center labels
        center_target = target[
            :,
            0,
            center_index,
            center_index
        ]

        all_center_labels.extend(
            center_target
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    center_probs = np.asarray(
        all_center_probs
    )

    center_labels = np.asarray(
        all_center_labels
    )

    try:

        roc_auc = roc_auc_score(
            center_labels,
            center_probs
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            center_labels,
            center_probs
        )

    except ValueError:

        pr_auc = float("nan")

    if len(all_tv_scores) > 0:

        mean_tv = float(
            np.mean(all_tv_scores)
        )

    else:

        mean_tv = 0.0

    return (
        total_loss / count,
        total_bce / count,
        mean_tv,
        roc_auc,
        pr_auc
    )


# ============================================================
# TRAIN
# ============================================================

print()
print("============================================================")
print("START TRAINING")
print("============================================================")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_score = -np.inf

history = []


for epoch in range(1, EPOCHS + 1):

    print()
    print(
        f"================ EPOCH {epoch}/{EPOCHS} ================"
    )

    train_loss, train_bce, train_tv = (
        train_one_epoch()
    )

    (
        val_loss,
        val_bce,
        val_tversky,
        val_roc,
        val_pr
    ) = validate()

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print(
        f"Train Loss       : {train_loss:.6f}"
    )

    print(
        f"Train BCE        : {train_bce:.6f}"
    )

    print(
        f"Train Tversky    : {train_tv:.6f}"
    )

    print(
        f"Validation Loss  : {val_loss:.6f}"
    )

    print(
        f"Validation BCE   : {val_bce:.6f}"
    )

    print(
        f"Validation TV    : {val_tversky:.6f}"
    )

    print(
        f"Center ROC-AUC   : {val_roc:.6f}"
    )

    print(
        f"Center PR-AUC    : {val_pr:.6f}"
    )

    print(
        f"Learning Rate    : {current_lr:.8f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_bce": train_bce,
        "train_tversky": train_tv,
        "val_loss": val_loss,
        "val_bce": val_bce,
        "val_tversky": val_tversky,
        "val_roc_auc": val_roc,
        "val_pr_auc": val_pr,
        "lr": current_lr
    })

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_tversky
    )

    # --------------------------------------------------------
    # Save best model
    #
    # We select based on the spatial Tversky objective rather
    # than ordinary validation BCE.
    # --------------------------------------------------------

    if val_tversky > best_score:

        best_score = val_tversky

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "val_tversky":
                    val_tversky,

                "val_roc_auc":
                    val_roc,

                "val_pr_auc":
                    val_pr,

                "patch_size":
                    PATCH_SIZE,

                "radius":
                    RADIUS,

                "alpha":
                    ALPHA,

                "beta":
                    BETA
            },
            BEST_MODEL
        )

        print()
        print(
            "NEW BEST MODEL SAVED"
        )

        print(
            f"Best validation Tversky: "
            f"{best_score:.6f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "epoch":
            EPOCHS,

        "patch_size":
            PATCH_SIZE,

        "radius":
            RADIUS,

        "alpha":
            ALPHA,

        "beta":
            BETA
    },
    FINAL_MODEL
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    "outputs/"
    "tversky_unet_training_history.npy"
)

os.makedirs(
    "outputs",
    exist_ok=True
)

np.save(
    history_path,
    history,
    allow_pickle=True
)


# ============================================================
# FINISHED
# ============================================================

print()
print("============================================================")
print("TRAINING COMPLETE")
print("============================================================")

print(
    f"Best validation Tversky: {best_score:.6f}"
)

print()
print(
    "Best model:"
)

print(
    BEST_MODEL
)

print()
print(
    "Final model:"
)

print(
    FINAL_MODEL
)

print()
print(
    "Next step:"
)

print(
    "Run tiled full-raster inference using the BEST model."
)

print(
    "Do NOT create the competition submission from only "
    "the 20,000 validation centers."
)