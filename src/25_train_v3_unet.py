import os
import random
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
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
    "fault_v3_unet_best.pt"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fault_v3_unet.pt"
)

SEED = 42

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

BATCH_SIZE = 16
EPOCHS = 8

LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-4

POS_WEIGHT = 4.0
FOCAL_GAMMA = 2.0

# Number of samples
N_POSITIVE = 44000
N_ORDINARY_NEGATIVE = 44000
N_NEAR_NEGATIVE = 22000

NUM_WORKERS = 0

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# SPATIAL BLOCK CONFIGURATION
# ============================================================

# Same 4x4 spatial split used earlier
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

        if br == 3:
            row_end = height
        else:
            row_end = (br + 1) * block_h

        if bc == 3:
            col_end = width
        else:
            col_end = (bc + 1) * block_w

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
# CONV BLOCK
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
# U-NET
# ============================================================

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

        # Handle possible spatial mismatch
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
# PATCH DATASET
# ============================================================

class FaultPatchDataset(Dataset):

    def __init__(
        self,
        coords,
        labels,
        feature_path,
        means,
        stds,
        patch_size=31
    ):

        self.coords = coords
        self.labels = labels

        self.feature_path = feature_path

        self.means = means.astype(
            np.float32
        )

        self.stds = stds.astype(
            np.float32
        )

        self.patch_size = patch_size

        self.radius = patch_size // 2

        self.src = None

    def _open(self):

        if self.src is None:

            self.src = rasterio.open(
                self.feature_path
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

        values = self.src.read(
            window=window
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        expected_shape = (
            19,
            self.patch_size,
            self.patch_size
        )

        if values.shape != expected_shape:

            raise RuntimeError(
                f"Invalid patch shape at "
                f"({row},{col}): "
                f"{values.shape}, "
                f"expected {expected_shape}"
            )

        # ----------------------------------------------------
        # Detect nodata / overflow values
        # ----------------------------------------------------

        invalid = (
            ~np.isfinite(values)
            |
            (np.abs(values) > 1e30)
        )

        # ----------------------------------------------------
        # Replace invalid values with band mean
        # ----------------------------------------------------

        for band in range(values.shape[0]):

            mask = invalid[band]

            if np.any(mask):

                values[band][mask] = self.means[band]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        values = (
            values - self.means[:, None, None]
        ) / (
            self.stds[:, None, None] + 1e-8
        )

        # ----------------------------------------------------
        # Clip extreme normalized values
        # ----------------------------------------------------

        values = np.clip(
            values,
            -10.0,
            10.0
        )

        # ----------------------------------------------------
        # Set originally invalid pixels to zero
        # ----------------------------------------------------

        invalid_any = np.any(
            invalid,
            axis=0
        )

        values[:, invalid_any] = 0.0

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        target = float(
            self.labels[idx]
        )

        x = torch.from_numpy(
            values
        )

        y = torch.tensor(
            target,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# FOCAL BCE
# ============================================================

def focal_bce_loss(
    logits,
    targets,
    gamma=2.0,
    pos_weight=4.0
):

    targets = targets.float()

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(
            pos_weight,
            device=logits.device
        )
    )

    probabilities = torch.sigmoid(
        logits
    )

    pt = torch.where(
        targets == 1,
        probabilities,
        1.0 - probabilities
    )

    focal_factor = (
        1.0 - pt
    ) ** gamma

    return (
        focal_factor * bce
    ).mean()


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    logits,
    targets
):

    logits = logits.view(-1)
    targets = targets.view(-1)

    # Weighted BCE

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(
            POS_WEIGHT,
            device=logits.device
        )
    )

    # Focal BCE

    focal = focal_bce_loss(
        logits,
        targets,
        gamma=FOCAL_GAMMA,
        pos_weight=POS_WEIGHT
    )

    loss = (
        0.45 * bce
        +
        0.55 * focal
    )

    return loss, bce.detach(), focal.detach()


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_focal = 0.0

    count = 0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(x)

        # U-Net produces a full patch.
        # We use the center pixel because the
        # dataset label corresponds to the center.

        center = output[
            :,
            0,
            RADIUS,
            RADIUS
        ]

        loss, bce, focal = combined_loss(
            center,
            y
        )

        loss.backward()

        # Gradient clipping for stability

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        batch_size = x.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_bce += (
            bce.item() * batch_size
        )

        total_focal += (
            focal.item() * batch_size
        )

        count += batch_size

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/{len(loader)} "
                f"Loss: "
                f"{loss.item():.4f}"
            )

    return (
        total_loss / count,
        total_bce / count,
        total_focal / count
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_sampled(
    model,
    loader
):

    model.eval()

    all_probs = []
    all_targets = []

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        output = model(x)

        center = output[
            :,
            0,
            RADIUS,
            RADIUS
        ]

        probs = torch.sigmoid(
            center
        )

        all_probs.append(
            probs.cpu().numpy()
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

    # ROC AUC

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
    print("V3 Tversky / Near-Fault U-NET")
    print("=" * 70)

    print(
        f"Using device: {DEVICE}"
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD NORMALIZATION
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

    print(
        "Training coordinates:",
        train_coords_all.shape
    )

    print(
        "Validation coordinates:",
        val_coords_all.shape
    )

    # ========================================================
    # OPEN RASTERS
    # ========================================================

    with rasterio.open(
        FEATURE_PATH
    ) as src:

        height = src.height
        width = src.width
        band_count = src.count

    with rasterio.open(
        LABEL_PATH
    ) as src:

        labels_full = src.read(
            1
        )

    print()
    print(
        "Raster:",
        height,
        "x",
        width
    )

    print(
        "Bands:",
        band_count
    )

    # ========================================================
    # SPATIAL MASKS
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING SPATIAL MASKS")
    print("=" * 70)

    train_spatial_mask, val_spatial_mask = \
        build_spatial_masks(
            height,
            width
        )

    print(
        "Training pixels:",
        train_spatial_mask.sum()
    )

    print(
        "Validation pixels:",
        val_spatial_mask.sum()
    )

    # ========================================================
    # SAFE COORDINATE FILTER
    # ========================================================

    def safe_coords(coords):

        coords = np.asarray(
            coords
        )

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
    # LABEL MASK
    # ========================================================

    fault_mask = (
        labels_full > 0
    )

    print()
    print(
        "Total fault pixels:",
        fault_mask.sum()
    )

    # ========================================================
    # IMPORTANT:
    # ONLY TRAINING-REGION FAULTS ARE USED TO BUILD
    # NEAR-FAULT NEGATIVES.
    #
    # This prevents validation labels from leaking
    # into training.
    # ========================================================

    training_fault_mask = (
        fault_mask
        &
        train_spatial_mask
    )

    print(
        "Training-region fault pixels:",
        training_fault_mask.sum()
    )

    # ========================================================
    # DISTANCE FROM TRAINING FAULTS
    # ========================================================

    print()
    print("=" * 70)
    print("BUILDING NEAR-FAULT NEGATIVE POOL")
    print("=" * 70)

    distance_map = distance_transform_edt(
        ~training_fault_mask
    )

    # Negative pixels within 3 pixels of a
    # TRAINING fault.

    near_negative_mask = (
        train_spatial_mask
        &
        (~fault_mask)
        &
        (distance_map <= 3)
        &
        (distance_map > 0)
    )

    near_coords = np.column_stack(
        np.where(
            near_negative_mask
        )
    )

    # Safe patch boundaries

    near_coords = safe_coords(
        near_coords
    )

    print(
        "Near-fault candidates:",
        len(near_coords)
    )

    # ========================================================
    # ORDINARY TRAINING COORDINATES
    # ========================================================

    # Keep only coordinates inside training spatial region

    train_in_region = []

    for row, col in train_coords_all:

        row = int(row)
        col = int(col)

        if train_spatial_mask[
            row,
            col
        ]:

            train_in_region.append(
                [row, col]
            )

    train_in_region = np.asarray(
        train_in_region,
        dtype=np.int32
    )

    # ========================================================
    # POSITIVE / NEGATIVE SPLIT
    # ========================================================

    positive_coords = []
    ordinary_negative_coords = []

    for row, col in train_in_region:

        if fault_mask[
            row,
            col
        ]:

            positive_coords.append(
                [row, col]
            )

        else:

            ordinary_negative_coords.append(
                [row, col]
            )

    positive_coords = np.asarray(
        positive_coords,
        dtype=np.int32
    )

    ordinary_negative_coords = np.asarray(
        ordinary_negative_coords,
        dtype=np.int32
    )

    print()
    print(
        "Positive coordinates:",
        len(positive_coords)
    )

    print(
        "Ordinary negatives:",
        len(ordinary_negative_coords)
    )

    print(
        "Near-fault negatives:",
        len(near_coords)
    )

    # ========================================================
    # SAMPLE TRAINING DATA
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    def sample_coords(
        coords,
        n
    ):

        if len(coords) <= n:

            return coords

        indices = rng.choice(
            len(coords),
            size=n,
            replace=False
        )

        return coords[
            indices
        ]

    positive_sample = sample_coords(
        positive_coords,
        N_POSITIVE
    )

    ordinary_negative_sample = sample_coords(
        ordinary_negative_coords,
        N_ORDINARY_NEGATIVE
    )

    near_negative_sample = sample_coords(
        near_coords,
        N_NEAR_NEGATIVE
    )

    # ========================================================
    # CREATE TRAINING COORDINATE ARRAY
    # ========================================================

    train_coords = np.concatenate(
        [
            positive_sample,
            ordinary_negative_sample,
            near_negative_sample
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
                len(ordinary_negative_sample),
                dtype=np.float32
            ),

            np.zeros(
                len(near_negative_sample),
                dtype=np.float32
            )
        ]
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    permutation = rng.permutation(
        len(train_coords)
    )

    train_coords = train_coords[
        permutation
    ]

    train_labels = train_labels[
        permutation
    ]

    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TRAINING DATA")
    print("=" * 70)

    print(
        "Positive:",
        len(positive_sample)
    )

    print(
        "Ordinary negative:",
        len(ordinary_negative_sample)
    )

    print(
        "Near-fault negative:",
        len(near_negative_sample)
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
        means,
        stds,
        PATCH_SIZE
    )

    # --------------------------------------------------------
    # Validation labels from actual validation coordinates
    # --------------------------------------------------------

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

    # Limit validation size to make training faster

    MAX_VAL = 40000

    if len(val_coords_all) > MAX_VAL:

        val_indices = rng.choice(
            len(val_coords_all),
            size=MAX_VAL,
            replace=False
        )

        val_coords = val_coords_all[
            val_indices
        ]

        val_labels = val_labels[
            val_indices
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
        means,
        stds,
        PATCH_SIZE
    )

    # ========================================================
    # DATALOADERS
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
    print("CREATING V3 U-NET")
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
    # TRAINING LOOP
    # ========================================================

    best_pr = -1.0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            f"========== EPOCH {epoch}/{EPOCHS} =========="
        )

        train_loss, train_bce, train_focal = \
            train_one_epoch(
                model,
                train_loader,
                optimizer
            )

        val_roc, val_pr = evaluate_sampled(
            model,
            val_loader
        )

        scheduler.step(
            val_pr
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print()
        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Train BCE  : {train_bce:.6f}"
        )

        print(
            f"Train Focal: {train_focal:.6f}"
        )

        print(
            f"Val ROC-AUC: {val_roc:.6f}"
        )

        print(
            f"Val PR-AUC : {val_pr:.6f}"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint based on PR-AUC
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
    # SAVE FINAL MODEL
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
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation PR-AUC:",
        best_pr
    )

    print()
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