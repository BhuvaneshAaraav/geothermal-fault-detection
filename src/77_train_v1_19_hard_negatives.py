import os
import random
import numpy as np
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG — EXACT V1 SETTINGS
# ============================================================

FEATURE_RASTER = "data/raw/training_features.tif"
LABEL_RASTER = "data/raw/Training_fault_labels.tif"

TRAIN_COORDS = "data/processed/unet/train_coords.npy"
TRAIN_LABELS = "data/processed/unet/train_labels.npy"

HARD_NEGATIVE_COORDS = (
    "data/processed/hard_negatives_v1_19/"
    "safe_hard_negative_coords.npy"
)

HARD_NEGATIVE_COUNT = 44086

VAL_COORDS = "data/processed/unet/val_coords.npy"
VAL_LABELS = "data/processed/unet/val_labels.npy"

MEANS_FILE = "data/processed/cnn/feature_means.npy"
STDS_FILE = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "outputs/final_test/reduced_feature_models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

PATCH_SIZE = 31
RADIUS = 3

BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 1e-3

BCE_WEIGHT = 0.50
TVERSKY_WEIGHT = 0.50

ALPHA = 0.2
BETA = 0.8

VAL_MAX_SAMPLES = 20000
GRAD_CLIP = 2.0


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg",
    "det_elev_slope",
]


# Remove one feature at a time
EXPERIMENTS = {
    "V1_19_HN": list(range(19)),
}


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
# DATASET — EXACT V1
# ============================================================

class FaultPatchDataset(Dataset):

    def __init__(
        self,
        feature_array,
        label_array,
        valid_array,
        coords,
        patch_size=31,
        feature_indices=None
    ):

        self.features = feature_array
        self.labels = label_array
        self.valid = valid_array

        self.coords = coords.astype(np.int32)

        self.patch_size = patch_size
        self.radius = patch_size // 2

        if feature_indices is None:
            feature_indices = list(range(feature_array.shape[0]))

        self.feature_indices = np.asarray(
            feature_indices,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        y, x = self.coords[idx]

        r = self.radius

        feature_patch = self.features[
            self.feature_indices,
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
# U-NET — EXACT V1
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

    def __init__(self, in_channels):

        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

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
# BCE — EXACT V1
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

    loss = loss * valid_mask

    denominator = valid_mask.sum().clamp_min(1.0)

    return loss.sum() / denominator


# ============================================================
# ZERO-PADDING SHIFT — EXACT V1
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
# DISTANCE TVERSKY — EXACT V1
# ============================================================

def distance_tversky_score(
    probabilities,
    targets,
    valid_mask,
    radius=3,
    alpha=0.2,
    beta=0.8
):

    margin = radius

    if probabilities.shape[-1] <= 2 * margin:
        raise ValueError(
            "Patch too small for selected radius."
        )

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

    tp_match = torch.zeros_like(p)
    gt_influence = torch.zeros_like(t)

    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            distance = np.sqrt(
                float(dx * dx + dy * dy)
            )

            if distance > radius:
                continue

            kernel = 1.0 - distance / radius

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

    tp_match = tp_match * valid
    gt_influence = gt_influence * valid

    tp = (
        t * tp_match
    ).sum(dim=(1, 2, 3))

    fn = (
        t * (1.0 - tp_match)
    ).sum(dim=(1, 2, 3))

    fp = (
        p * (1.0 - gt_influence) * valid
    ).sum(dim=(1, 2, 3))

    denominator = (
        tp
        + alpha * fp
        + beta * fn
        + 1e-7
    )

    score = tp / denominator

    has_gt = (
        t.sum(dim=(1, 2, 3)) > 0
    )

    return score, has_gt


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

with rasterio.open(FEATURE_RASTER) as src:

    features = src.read().astype(np.float32)

    feature_mask = (
        src.read_masks(1) > 0
    )

with rasterio.open(LABEL_RASTER) as src:

    labels = src.read(1).astype(np.float32)

valid = feature_mask.astype(np.float32)


# ============================================================
# EXACT V1 NORMALIZATION
# ============================================================

means = np.load(MEANS_FILE).astype(np.float32)
stds = np.load(STDS_FILE).astype(np.float32)

print("Feature shape:", features.shape)
print("Means shape:", means.shape)
print("Stds shape :", stds.shape)

for i in range(features.shape[0]):

    band = features[i]

    invalid = (
        (~feature_mask)
        |
        (~np.isfinite(band))
    )

    band[invalid] = means[i]

    band -= means[i]

    band /= max(
        float(stds[i]),
        1e-8
    )

    band[:] = np.clip(
        band,
        -10.0,
        10.0
    )

features[:, ~feature_mask] = 0.0


# ============================================================
# LOAD COORDINATES
# ============================================================

train_coords = np.load(
    TRAIN_COORDS
)

train_labels = np.load(
    TRAIN_LABELS
)

# --------------------------------------------------------
# V1_19 HARD NEGATIVES
# --------------------------------------------------------

hard_negative_coords_all = np.load(
    HARD_NEGATIVE_COORDS
)

if len(hard_negative_coords_all) < HARD_NEGATIVE_COUNT:
    raise ValueError(
        f"Only {len(hard_negative_coords_all)} safe hard negatives "
        f"available, but {HARD_NEGATIVE_COUNT} requested."
    )

hard_rng = np.random.default_rng(SEED)

hard_idx = hard_rng.choice(
    len(hard_negative_coords_all),
    HARD_NEGATIVE_COUNT,
    replace=False
)

hard_negative_coords = hard_negative_coords_all[hard_idx]

hard_negative_labels = np.zeros(
    HARD_NEGATIVE_COUNT,
    dtype=train_labels.dtype
)

original_train_count = len(train_coords)

train_coords = np.concatenate(
    [
        train_coords,
        hard_negative_coords
    ],
    axis=0
)

train_labels = np.concatenate(
    [
        train_labels,
        hard_negative_labels
    ],
    axis=0
)

hard_negative_scores_all = np.load(
    "data/processed/hard_negatives_v1_19/"
    "safe_hard_negative_scores.npy"
)

selected_hard_scores = hard_negative_scores_all[hard_idx]

print(
    "Original training coordinates:",
    original_train_count
)

print(
    "Added hard negatives:",
    len(hard_negative_coords)
)

print(
    "Combined training coordinates:",
    train_coords.shape
)

print(
    "Hard-negative score range:",
    float(selected_hard_scores.min()),
    "to",
    float(selected_hard_scores.max())
)

val_coords = np.load(
    VAL_COORDS
)

val_labels = np.load(
    VAL_LABELS
)

print("Training coordinates:", train_coords.shape)
print("Validation coordinates:", val_coords.shape)


# ============================================================
# EXACT V1 VALIDATION SUBSET
# ============================================================

if len(val_coords) > VAL_MAX_SAMPLES:

    rng = np.random.default_rng(SEED)

    val_idx = rng.choice(
        len(val_coords),
        VAL_MAX_SAMPLES,
        replace=False
    )

    val_coords_eval = val_coords[val_idx]
    val_labels_eval = val_labels[val_idx]

else:

    val_coords_eval = val_coords
    val_labels_eval = val_labels


# ============================================================
# TRAIN ONE EXPERIMENT
# ============================================================

def train_experiment(name, feature_indices):

    print("\n")
    print("=" * 70)
    print("EXPERIMENT:", name)
    print("=" * 70)

    print("Input bands:", len(feature_indices))

    print(
        "Bands:",
        ", ".join(
            FEATURE_NAMES[i]
            for i in feature_indices
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = FaultPatchDataset(
        features,
        labels,
        valid,
        train_coords,
        PATCH_SIZE,
        feature_indices
    )

    val_dataset = FaultPatchDataset(
        features,
        labels,
        valid,
        val_coords_eval,
        PATCH_SIZE,
        feature_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = FaultSegmentationUNet(
        in_channels=len(feature_indices)
    ).to(DEVICE)

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

    best_tversky = -np.inf
    best_epoch = -1

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"{name}_best.pt"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:

            x, y, valid_mask = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)

            probabilities = torch.sigmoid(
                logits
            )

            bce = masked_bce_loss(
                logits,
                y,
                valid_mask
            )

            tv_score, has_gt = (
                distance_tversky_score(
                    probabilities,
                    y,
                    valid_mask,
                    radius=RADIUS,
                    alpha=ALPHA,
                    beta=BETA
                )
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

            loss = (
                BCE_WEIGHT * bce
                +
                TVERSKY_WEIGHT * tv_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

            train_loss_sum += (
                float(loss.detach().cpu())
            )

            train_batches += 1

        train_loss = (
            train_loss_sum /
            max(train_batches, 1)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_tversky_values = []

        center_probs = []
        center_targets = []

        with torch.no_grad():

            for batch in val_loader:

                x, y, valid_mask = batch

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                valid_mask = valid_mask.to(DEVICE)

                logits = model(x)

                probabilities = torch.sigmoid(
                    logits
                )

                tv_score, has_gt = (
                    distance_tversky_score(
                        probabilities,
                        y,
                        valid_mask,
                        radius=RADIUS,
                        alpha=ALPHA,
                        beta=BETA
                    )
                )

                if has_gt.any():

                    val_tversky_values.extend(
                        tv_score[
                            has_gt
                        ].detach().cpu().numpy()
                    )

                center = PATCH_SIZE // 2

                cp = probabilities[
                    :,
                    0,
                    center,
                    center
                ]

                ct = y[
                    :,
                    0,
                    center,
                    center
                ]

                center_probs.extend(
                    cp.detach().cpu().numpy()
                )

                center_targets.extend(
                    ct.detach().cpu().numpy()
                )

        val_tversky = (
            float(np.mean(val_tversky_values))
            if val_tversky_values
            else 0.0
        )

        center_probs = np.asarray(
            center_probs
        )

        center_targets = np.asarray(
            center_targets
        )

        try:

            val_roc = roc_auc_score(
                center_targets,
                center_probs
            )

        except ValueError:

            val_roc = float("nan")

        try:

            val_pr = average_precision_score(
                center_targets,
                center_probs
            )

        except ValueError:

            val_pr = float("nan")

        scheduler.step(
            val_tversky
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_loss:.6f} | "
            f"tversky={val_tversky:.6f} | "
            f"ROC-AUC={val_roc:.6f} | "
            f"PR-AUC={val_pr:.6f} | "
            f"lr={lr:.2e}"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if val_tversky > best_tversky:

            best_tversky = val_tversky
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_tversky":
                        val_tversky,
                    "val_roc_auc":
                        val_roc,
                    "val_pr_auc":
                        val_pr,
                    "feature_indices":
                        feature_indices,
                    "feature_names":
                        [
                            FEATURE_NAMES[i]
                            for i in feature_indices
                        ],
                    "patch_size":
                        PATCH_SIZE,
                    "radius":
                        RADIUS,
                    "alpha":
                        ALPHA,
                    "beta":
                        BETA,
                },
                checkpoint_path
            )

    print(
        f"\nBEST {name}: "
        f"Tversky={best_tversky:.6f} "
        f"at epoch {best_epoch}"
    )

    return best_tversky


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}

for name, feature_indices in EXPERIMENTS.items():

    # Reset RNG exactly before each experiment.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    results[name] = train_experiment(
        name,
        feature_indices
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for name, score in results.items():

    print(
        f"{name:10s} "
        f"best_val_tversky = {score:.6f}"
    )

print("\nCheckpoints saved in:")
print(MODEL_DIR)
