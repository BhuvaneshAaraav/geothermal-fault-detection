import os
import random
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

TRAIN_COORDS_PATH = "data/processed/unet/train_coords.npy"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fault_v4_unet_best.pt"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fault_v4_unet.pt"
)

SEED = 42

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

BATCH_SIZE = 16
EPOCHS = 8

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Spatial loss parameters
ALPHA = 0.2
BETA = 0.8
DISTANCE_RADIUS = 3

# Focal parameters
FOCAL_GAMMA = 2.0
# Number of training centers
N_POSITIVE = 44000
N_NEGATIVE = 44000

NUM_WORKERS = 0

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# SPATIAL BLOCKS
# ============================================================

TRAIN_BLOCKS = [
    0, 2, 3,
    4, 5, 6,
    8,
    11,
    12, 13, 14, 15
]

VAL_BLOCKS = [
    1, 7, 9, 10
]


# ============================================================
# BUILD SPATIAL MASK
# ============================================================

def build_spatial_masks(height, width):

    train_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    val_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    block_h = height // 4
    block_w = width // 4

    for block_id in range(16):

        br = block_id // 4
        bc = block_id % 4

        row_start = br * block_h
        col_start = bc * block_w

        row_end = (
            height
            if br == 3
            else (br + 1) * block_h
        )

        col_end = (
            width
            if bc == 3
            else (bc + 1) * block_w
        )

        if block_id in TRAIN_BLOCKS:

            train_mask[
                row_start:row_end,
                col_start:col_end
            ] = True

        elif block_id in VAL_BLOCKS:

            val_mask[
                row_start:row_end,
                col_start:col_end
            ] = True

    return train_mask, val_mask


# ============================================================
# MODEL
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

    def __init__(self):

        super().__init__()

        # Encoder

        self.enc1 = ConvBlock(
            19,
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
# PATCH DATASET
# ============================================================

class FaultPatchDataset(Dataset):

    def __init__(
        self,
        coords,
        labels,
        feature_path,
        label_path,
        means,
        stds,
        patch_size=31
    ):

        self.coords = coords
        self.labels = labels

        self.feature_path = feature_path
        self.label_path = label_path

        self.means = means.astype(
            np.float32
        )

        self.stds = stds.astype(
            np.float32
        )

        self.patch_size = patch_size
        self.radius = patch_size // 2

        self.feature_src = None
        self.label_src = None

    def _open(self):

        if self.feature_src is None:

            self.feature_src = rasterio.open(
                self.feature_path
            )

        if self.label_src is None:

            self.label_src = rasterio.open(
                self.label_path
            )

    def __len__(self):

        return len(self.coords)

    def __getitem__(self, idx):

        self._open()

        row, col = self.coords[idx]

        row = int(row)
        col = int(col)

        window = rasterio.windows.Window(
            col - self.radius,
            row - self.radius,
            self.patch_size,
            self.patch_size
        )

        features = self.feature_src.read(
            window=window
        ).astype(
            np.float32
        )

        labels = self.label_src.read(
            1,
            window=window
        )

        expected_shape = (
            19,
            self.patch_size,
            self.patch_size
        )

        if features.shape != expected_shape:

            raise RuntimeError(
                f"Invalid feature patch at "
                f"({row},{col}): "
                f"{features.shape}; "
                f"expected {expected_shape}"
            )

        if labels.shape != (
            self.patch_size,
            self.patch_size
        ):

            raise RuntimeError(
                f"Invalid label patch at "
                f"({row},{col}): "
                f"{labels.shape}"
            )

        # ----------------------------------------------------
        # Detect invalid raster values
        # ----------------------------------------------------

        invalid = (
            ~np.isfinite(features)
            |
            (np.abs(features) > 1e30)
        )

        # ----------------------------------------------------
        # Replace invalid values with band means
        # ----------------------------------------------------

        for band in range(19):

            bad = invalid[band]

            if np.any(bad):

                features[band][bad] = \
                    self.means[band]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        features = (
            features
            -
            self.means[:, None, None]
        ) / (
            self.stds[:, None, None]
            +
            1e-8
        )

        # ----------------------------------------------------
        # Clip
        # ----------------------------------------------------

        features = np.clip(
            features,
            -10.0,
            10.0
        )

        # ----------------------------------------------------
        # Invalid pixels become zero
        # ----------------------------------------------------

        invalid_any = np.any(
            invalid,
            axis=0
        )

        features[:, invalid_any] = 0.0

        # ----------------------------------------------------
        # Binary target
        # ----------------------------------------------------

        target = (
            labels > 0
        ).astype(
            np.float32
        )

        x = torch.from_numpy(
            features
        )

        y = torch.from_numpy(
            target
        ).unsqueeze(
            0
        )

        return x, y


# ============================================================
# DISTANCE WEIGHTS
# ============================================================

def make_distance_weights(
    target,
    radius=3
):

    """
    target:
        [B, 1, H, W]

    Returns weights where:

        distance 0 -> 1
        distance 1 -> 0.75
        distance 2 -> 0.50
        distance 3 -> 0.25
        farther   -> 0

    These weights are used for the fault-region
    TP component.
    """

    target_np = (
        target.detach()
        .cpu()
        .numpy()
    )

    weights = np.zeros_like(
        target_np,
        dtype=np.float32
    )

    for i in range(
        target_np.shape[0]
    ):

        gt = (
            target_np[i, 0] > 0.5
        )

        distance = distance_transform_edt(
            ~gt
        )

        nearby = (
            distance <= radius
        )

        weights[
            i,
            0
        ][nearby] = (
            1.0
            -
            distance[nearby]
            /
            (radius + 1.0)
        )

    return torch.from_numpy(
        weights
    ).to(
        target.device
    )


# ============================================================
# SPATIAL TVERSKY LOSS
# ============================================================

def spatial_tversky_loss(
    logits,
    target
):

    probability = torch.sigmoid(
        logits
    )

    # Distance-aware TP weighting

    distance_weights = make_distance_weights(
        target,
        DISTANCE_RADIUS
    )

    weighted_tp = torch.sum(
        probability
        *
        target
        *
        distance_weights,
        dim=(1, 2, 3)
    )

    # False positives

    weighted_fp = torch.sum(
        probability
        *
        (1.0 - target),
        dim=(1, 2, 3)
    )

    # False negatives

    weighted_fn = torch.sum(
        (1.0 - probability)
        *
        target,
        dim=(1, 2, 3)
    )

    score = (
        weighted_tp
        /
        (
            weighted_tp
            +
            ALPHA * weighted_fp
            +
            BETA * weighted_fn
            +
            1e-6
        )
    )

    return (
        1.0
        -
        score
    ).mean()


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    targets,
    gamma=2.0
):

    probability = torch.sigmoid(
        logits
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    pt = torch.where(
        targets > 0.5,
        probability,
        1.0 - probability
    )

    focal = (
        (1.0 - pt) ** gamma
    ) * bce

    return focal.mean()


# ============================================================
# COMBINED V4 LOSS
# ============================================================

def v4_loss(
    logits,
    target
):

    # Spatial Tversky

    tversky = spatial_tversky_loss(
        logits,
        target
    )

    # Focal segmentation loss

    focal = focal_loss(
        logits,
        target,
        FOCAL_GAMMA
    )

    # Standard BCE with mild positive weighting

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(
            2.0,
            device=logits.device
        )
    )

    loss = (
        0.55 * tversky
        +
        0.30 * focal
        +
        0.15 * bce
    )

    return (
        loss,
        tversky.detach(),
        focal.detach(),
        bce.detach()
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_tversky = 0.0
    total_focal = 0.0
    total_bce = 0.0

    count = 0

    for batch_idx, (
        x,
        target
    ) in enumerate(loader):

        x = x.to(
            DEVICE
        )

        target = target.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss, tversky, focal, bce = \
            v4_loss(
                logits,
                target
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        batch_size = x.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_tversky += (
            tversky.item()
            *
            batch_size
        )

        total_focal += (
            focal.item()
            *
            batch_size
        )

        total_bce += (
            bce.item()
            *
            batch_size
        )

        count += batch_size

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)} "
                f"Loss: "
                f"{loss.item():.4f}"
            )

    return (
        total_loss / count,
        total_tversky / count,
        total_focal / count,
        total_bce / count
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_validation(
    model,
    loader
):

    model.eval()

    all_probs = []
    all_targets = []

    for x, target in loader:

        x = x.to(
            DEVICE
        )

        logits = model(x)

        # Center prediction is used ONLY for
        # quick PR-AUC model selection.
        #
        # Full competition evaluation will use
        # the complete raster later.

        center = logits[
            :,
            0,
            RADIUS,
            RADIUS
        ]

        probability = torch.sigmoid(
            center
        )

        all_probs.append(
            probability
            .cpu()
            .numpy()
        )

        all_targets.append(
            target[
                :,
                0,
                RADIUS,
                RADIUS
            ]
            .cpu()
            .numpy()
        )

    probs = np.concatenate(
        all_probs
    )

    targets = np.concatenate(
        all_targets
    )

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score
    )

    try:

        roc = roc_auc_score(
            targets,
            probs
        )

    except Exception:

        roc = 0.0

    try:

        pr = average_precision_score(
            targets,
            probs
        )

    except Exception:

        pr = 0.0

    return roc, pr


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V4 SPATIAL TVERSKY U-NET")
    print("=" * 70)

    print(
        "Using device:",
        DEVICE
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)

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

    print(
        "Means shape:",
        means.shape
    )

    print(
        "Stds shape :",
        stds.shape
    )

    # ========================================================
    # LOAD COORDINATES
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING COORDINATES")
    print("=" * 70)

    train_coords_all = np.load(
        TRAIN_COORDS_PATH
    )

    val_coords_all = np.load(
        VAL_COORDS_PATH
    )

    # ========================================================
    # RASTER SIZE
    # ========================================================

    with rasterio.open(
        FEATURE_PATH
    ) as src:

        height = src.height
        width = src.width
        bands = src.count

    with rasterio.open(
        LABEL_PATH
    ) as src:

        labels_full = src.read(
            1
        )

    print(
        "Raster:",
        height,
        "x",
        width
    )

    print(
        "Bands:",
        bands
    )

    # ========================================================
    # SPATIAL MASKS
    # ========================================================

    train_mask, val_mask = \
        build_spatial_masks(
            height,
            width
        )

    print()
    print(
        "Training pixels:",
        int(train_mask.sum())
    )

    print(
        "Validation pixels:",
        int(val_mask.sum())
    )

    # ========================================================
    # SAFE COORDINATES
    # ========================================================

    def safe_coords(coords):

        mask = (
            (coords[:, 0] >= RADIUS)
            &
            (coords[:, 0] < height - RADIUS)
            &
            (coords[:, 1] >= RADIUS)
            &
            (coords[:, 1] < width - RADIUS)
        )

        return coords[mask]

    train_coords_all = safe_coords(
        train_coords_all
    )

    val_coords_all = safe_coords(
        val_coords_all
    )

    # ========================================================
    # FILTER TRAINING REGION
    # ========================================================

    train_coords_all = np.asarray(
        [
            [r, c]
            for r, c in train_coords_all
            if train_mask[
                int(r),
                int(c)
            ]
        ],
        dtype=np.int32
    )

    print()
    print(
        "Safe training coordinates:",
        len(train_coords_all)
    )

    print(
        "Safe validation coordinates:",
        len(val_coords_all)
    )

    # ========================================================
    # FAULT MASK
    # ========================================================

    fault_mask = (
        labels_full > 0
    )

    print()
    print(
        "Total fault pixels:",
        int(fault_mask.sum())
    )

    print(
        "Training fault pixels:",
        int(
            (
                fault_mask
                &
                train_mask
            ).sum()
        )
    )

    # ========================================================
    # POSITIVE / NEGATIVE CENTERS
    # ========================================================

    positive_coords = []
    negative_coords = []

    for row, col in train_coords_all:

        row = int(row)
        col = int(col)

        if fault_mask[
            row,
            col
        ]:

            positive_coords.append(
                [row, col]
            )

        else:

            negative_coords.append(
                [row, col]
            )

    positive_coords = np.asarray(
        positive_coords,
        dtype=np.int32
    )

    negative_coords = np.asarray(
        negative_coords,
        dtype=np.int32
    )

    print()
    print(
        "Positive centers:",
        len(positive_coords)
    )

    print(
        "Negative centers:",
        len(negative_coords)
    )

    # ========================================================
    # SAMPLE BALANCED CENTERS
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    def sample(
        coords,
        n
    ):

        n = min(
            n,
            len(coords)
        )

        indices = rng.choice(
            len(coords),
            size=n,
            replace=False
        )

        return coords[
            indices
        ]

    positive_sample = sample(
        positive_coords,
        N_POSITIVE
    )

    negative_sample = sample(
        negative_coords,
        N_NEGATIVE
    )

    train_coords = np.concatenate(
        [
            positive_sample,
            negative_sample
        ],
        axis=0
    )

    train_labels = np.concatenate(
        [
            np.ones(
                len(positive_sample),
                dtype=np.float32
            ),
            np.zeros(
                len(negative_sample),
                dtype=np.float32
            )
        ]
    )

    permutation = rng.permutation(
        len(train_coords)
    )

    train_coords = train_coords[
        permutation
    ]

    train_labels = train_labels[
        permutation
    ]

    print()
    print("=" * 70)
    print("TRAINING DATA")
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
        len(train_coords)
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = FaultPatchDataset(
        train_coords,
        train_labels,
        FEATURE_PATH,
        LABEL_PATH,
        means,
        stds,
        PATCH_SIZE
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_labels = np.array(
        [
            float(
                fault_mask[
                    int(row),
                    int(col)
                ]
            )
            for row, col in val_coords_all
        ],
        dtype=np.float32
    )

    MAX_VAL = 40000

    if len(val_coords_all) > MAX_VAL:

        indices = rng.choice(
            len(val_coords_all),
            size=MAX_VAL,
            replace=False
        )

        val_coords = val_coords_all[
            indices
        ]

        val_labels = val_labels[
            indices
        ]

    else:

        val_coords = val_coords_all

    print()
    print(
        "Validation samples:",
        len(val_coords)
    )

    print(
        "Validation positives:",
        int(val_labels.sum())
    )

    print(
        "Validation negatives:",
        int(
            len(val_labels)
            -
            val_labels.sum()
        )
    )

    val_dataset = FaultPatchDataset(
        val_coords,
        val_labels,
        FEATURE_PATH,
        LABEL_PATH,
        means,
        stds,
        PATCH_SIZE
    )

    # ========================================================
    # LOADERS
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING V4 U-NET")
    print("=" * 70)

    model = FaultSegmentationUNet().to(
        DEVICE
    )

    print(model)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_pr = -1.0

    print()
    print("=" * 70)
    print("STARTING V4 TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            f"========== EPOCH "
            f"{epoch}/{EPOCHS} =========="
        )

        (
            train_loss,
            train_tversky,
            train_focal,
            train_bce
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_roc, val_pr = \
            evaluate_validation(
                model,
                val_loader
            )

        scheduler.step(
            val_pr
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print()
        print(
            f"Train Loss    : "
            f"{train_loss:.6f}"
        )

        print(
            f"Train Tversky : "
            f"{train_tversky:.6f}"
        )

        print(
            f"Train Focal   : "
            f"{train_focal:.6f}"
        )

        print(
            f"Train BCE     : "
            f"{train_bce:.6f}"
        )

        print(
            f"Val ROC-AUC   : "
            f"{val_roc:.6f}"
        )

        print(
            f"Val PR-AUC    : "
            f"{val_pr:.6f}"
        )

        print(
            f"Learning rate : "
            f"{lr:.8f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_pr > best_pr:

            best_pr = val_pr

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_roc":
                        val_roc,

                    "val_pr":
                        val_pr
                },
                BEST_MODEL_PATH
            )

            print()
            print(
                ">>> SAVED BEST MODEL"
            )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epoch":
                EPOCHS,

            "best_val_pr":
                best_pr
        },
        FINAL_MODEL_PATH
    )

    print()
    print("=" * 70)
    print("V4 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation PR-AUC:",
        best_pr
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print(
        "Final model:",
        FINAL_MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()