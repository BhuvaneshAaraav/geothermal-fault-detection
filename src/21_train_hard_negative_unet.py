import os
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

HARD_COORDS_PATH = (
    "data/processed/hard_negatives/"
    "hard_negative_coords.npy"
)

HARD_SCORES_PATH = (
    "data/processed/hard_negatives/"
    "hard_negative_scores.npy"
)

MEAN_PATH = (
    "data/processed/cnn/"
    "feature_means.npy"
)

STD_PATH = (
    "data/processed/cnn/"
    "feature_stds.npy"
)

BEST_MODEL_PATH = (
    "models/fault_hard_negative_unet_best.pt"
)

FINAL_MODEL_PATH = (
    "models/fault_hard_negative_unet.pt"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

POSITIVE_COUNT = 44_000
ORDINARY_NEGATIVE_COUNT = 44_000

HARD_NEGATIVE_COUNT = 22_000
MODERATE_HARD_COUNT = 22_000

BATCH_SIZE = 16

EPOCHS = 8

LEARNING_RATE = 1e-3

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
# CONV BLOCK
# EXACTLY MATCHES src/14_train_tversky_unet.py
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


# ============================================================
# REAL PATCH SEGMENTATION U-NET
# EXACTLY MATCHES MODEL 14
# ============================================================

class FaultSegmentationUNet(nn.Module):

    def __init__(
        self,
        in_channels=19
    ):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(
            self.pool3(e3)
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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
# DATASET
# ============================================================

class FaultPatchDataset(
    Dataset
):

    def __init__(
        self,
        features,
        labels,
        feature_mask,
        coordinates,
        means,
        stds
    ):

        self.features = features

        self.labels = labels

        self.feature_mask = feature_mask

        self.coordinates = coordinates

        self.means = means

        self.stds = stds

        self.radius = PATCH_SIZE // 2


    def __len__(self):

        return len(
            self.coordinates
        )


    def __getitem__(
        self,
        index
    ):

        y, x = self.coordinates[
            index
        ]

        y = int(y)
        x = int(x)

        r = self.radius

        # ----------------------------------------------------
        # Extract feature patch
        # ----------------------------------------------------

        patch = self.features[
            :,
            y-r:y+r+1,
            x-r:x+r+1
        ].copy()


        # ----------------------------------------------------
        # Extract label patch
        # ----------------------------------------------------

        label_patch = self.labels[
            y-r:y+r+1,
            x-r:x+r+1
        ].copy()


        # ----------------------------------------------------
        # Extract raster validity mask
        # ----------------------------------------------------

        patch_valid = self.feature_mask[
            y-r:y+r+1,
            x-r:x+r+1
        ]


        # ----------------------------------------------------
        # CRITICAL PREPROCESSING
        #
        # Invalid values MUST be replaced BEFORE
        # normalization.
        #
        # This prevents:
        #
        # inf / nan / extreme values
        #
        # from causing overflow.
        # ----------------------------------------------------

        for band in range(
            patch.shape[0]
        ):

            values = patch[
                band
            ]

            invalid = (
                ~patch_valid
                |
                ~np.isfinite(values)
                |
                (np.abs(values) > 1e30)
            )

            values[invalid] = (
                self.means[band]
            )

            patch[
                band
            ] = values


        # ----------------------------------------------------
        # Normalize
        #
        # IMPORTANT:
        # use broadcasting across H/W.
        # ----------------------------------------------------

        patch = (
            patch
            -
            self.means[
                :,
                None,
                None
            ]
        ) / (
            self.stds[
                :,
                None,
                None
            ]
        )


        # ----------------------------------------------------
        # Clip normalized features
        # Same preprocessing as Model 14.
        # ----------------------------------------------------

        patch = np.clip(
            patch,
            -10.0,
            10.0
        )


        # ----------------------------------------------------
        # Explicitly zero invalid pixels
        # ----------------------------------------------------

        patch[
            :,
            ~patch_valid
        ] = 0.0


        # ----------------------------------------------------
        # Final finite-value safety check
        # ----------------------------------------------------

        patch[
            ~np.isfinite(patch)
        ] = 0.0


        patch = patch.astype(
            np.float32
        )


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        label_patch = (
            label_patch > 0
        ).astype(
            np.float32
        )


        return (
            torch.from_numpy(
                patch
            ),

            torch.from_numpy(
                label_patch
            ).unsqueeze(0)
        )


# ============================================================
# LOAD RASTERS
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


with rasterio.open(
    FEATURES_PATH
) as src:

    features = src.read(
        out_dtype="float32"
    )

    feature_mask = (
        src.read_masks(1) > 0
    )

    height = src.height

    width = src.width


with rasterio.open(
    LABELS_PATH
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
# RASTER VALUE CHECK
# ============================================================

print()
print("=" * 70)
print("CHECKING RASTER VALUES")
print("=" * 70)


total_values = features.size

finite_values = np.isfinite(
    features
).sum()


print(
    "Total values:",
    total_values
)

print(
    "Finite values:",
    finite_values
)

print(
    "Non-finite values:",
    total_values - finite_values
)


for band in range(
    features.shape[0]
):

    band_data = features[
        band
    ]

    finite = band_data[
        np.isfinite(
            band_data
        )
    ]

    if len(finite) > 0:

        print(
            f"Band {band + 1:2d}: "
            f"min={finite.min():.6g}, "
            f"max={finite.max():.6g}"
        )


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print()
print("=" * 70)
print("LOADING NORMALIZATION")
print("=" * 70)


means = np.load(
    MEAN_PATH
).astype(
    np.float32
)


stds = np.load(
    STD_PATH
).astype(
    np.float32
)


# Prevent division by zero.

stds = np.maximum(
    stds,
    1e-6
)


print(
    "Means shape:",
    means.shape
)

print(
    "Stds shape:",
    stds.shape
)


# ============================================================
# BUILD SPATIAL TRAINING MASK
# ============================================================

print()
print("=" * 70)
print("BUILDING SPATIAL TRAINING MASK")
print("=" * 70)


rows_per_block = height // 4

cols_per_block = width // 4


# NEVER use these blocks for training.

validation_blocks = {
    1,
    7,
    9,
    10
}


training_mask = np.zeros(
    (height, width),
    dtype=bool
)


for block_id in range(16):

    block_row = block_id // 4

    block_col = block_id % 4


    y0 = (
        block_row
        * rows_per_block
    )


    y1 = (
        (block_row + 1)
        * rows_per_block
        if block_row < 3
        else height
    )


    x0 = (
        block_col
        * cols_per_block
    )


    x1 = (
        (block_col + 1)
        * cols_per_block
        if block_col < 3
        else width
    )


    if block_id not in validation_blocks:

        training_mask[
            y0:y1,
            x0:x1
        ] = True


print(
    "Training pixels:",
    int(
        training_mask.sum()
    )
)


print(
    "Validation pixels:",
    int(
        (~training_mask).sum()
    )
)


# ============================================================
# PATCH-SAFE TRAINING CENTERS
# ============================================================

print()
print("=" * 70)
print("CREATING SAFE TRAINING CENTERS")
print("=" * 70)


safe_mask = (
    training_mask.copy()
)


# Prevent 31x31 patches from
# going outside the raster.

safe_mask[
    :RADIUS,
    :
] = False


safe_mask[
    -RADIUS:,
    :
] = False


safe_mask[
    :,
    :RADIUS
] = False


safe_mask[
    :,
    -RADIUS:
] = False


# Make sure patch doesn't cross
# into validation blocks.

distance_from_validation = (
    distance_transform_edt(
        training_mask
    )
)


safe_mask &= (
    distance_from_validation
    > RADIUS
)


print(
    "Safe centers:",
    int(
        safe_mask.sum()
    )
)


# ============================================================
# POSITIVE CENTERS
# ============================================================

print()
print("=" * 70)
print("FINDING POSITIVE CENTERS")
print("=" * 70)


positive_mask = (
    safe_mask
    &
    feature_mask
    &
    (labels > 0)
)


positive_coords = np.argwhere(
    positive_mask
)


print(
    "Positive centers:",
    len(
        positive_coords
    )
)


# ============================================================
# ORDINARY NEGATIVES
# ============================================================

ordinary_negative_mask = (
    safe_mask
    &
    feature_mask
    &
    (labels == 0)
)


ordinary_negative_coords = np.argwhere(
    ordinary_negative_mask
)


print(
    "Ordinary negative candidates:",
    len(
        ordinary_negative_coords
    )
)


# ============================================================
# LOAD HARD NEGATIVES
# ============================================================

print()
print("=" * 70)
print("LOADING HARD NEGATIVES")
print("=" * 70)


hard_coords = np.load(
    HARD_COORDS_PATH
)


hard_scores = np.load(
    HARD_SCORES_PATH
)


print(
    "Hard-negative pool:",
    len(hard_coords)
)


# ============================================================
# VERIFY HARD NEGATIVES
# ============================================================

# Only keep coordinates that are actually
# in the safe training region and are labeled zero.
#
# This is an additional safety check.

valid_hard = []


for coord in hard_coords:

    y = int(
        coord[0]
    )

    x = int(
        coord[1]
    )

    if (
        0 <= y < height
        and
        0 <= x < width
        and
        safe_mask[y, x]
        and
        feature_mask[y, x]
        and
        labels[y, x] == 0
    ):

        valid_hard.append(
            [y, x]
        )


hard_coords = np.asarray(
    valid_hard,
    dtype=np.int32
)


print(
    "Verified hard negatives:",
    len(hard_coords)
)


# ============================================================
# SPLIT HARD NEGATIVES INTO SCORE BANDS
# ============================================================

# Recalculate scores for the retained
# coordinates directly from the probability
# map is not possible here, so instead use
# the original scores only if lengths match.
#
# Since Step 20 already verified these coordinates,
# we keep the original ordering by creating
# a coordinate -> score mapping.

original_coords = np.load(
    HARD_COORDS_PATH
)

original_scores = np.load(
    HARD_SCORES_PATH
)


score_lookup = {}

for coord, score in zip(
    original_coords,
    original_scores
):

    key = (
        int(coord[0]),
        int(coord[1])
    )

    score_lookup[key] = float(
        score
    )


verified_scores = np.array(
    [
        score_lookup[
            (
                int(coord[0]),
                int(coord[1])
            )
        ]
        for coord in hard_coords
    ],
    dtype=np.float32
)


# ------------------------------------------------------------
# Score bands
# ------------------------------------------------------------

very_hard = hard_coords[
    verified_scores >= 0.90
]


hard = hard_coords[
    (
        (verified_scores >= 0.80)
        &
        (verified_scores < 0.90)
    )
]


moderate = hard_coords[
    (
        (verified_scores >= 0.70)
        &
        (verified_scores < 0.80)
    )
]


easy_hard = hard_coords[
    (
        (verified_scores >= 0.60)
        &
        (verified_scores < 0.70)
    )
]


print(
    ">= 0.90:",
    len(very_hard)
)


print(
    "0.80 - 0.90:",
    len(hard)
)


print(
    "0.70 - 0.80:",
    len(moderate)
)


print(
    "0.60 - 0.70:",
    len(easy_hard)
)


# ============================================================
# SAMPLING FUNCTION
# ============================================================

def sample_coords(
    coords,
    count
):

    if len(coords) == 0:

        return np.empty(
            (0, 2),
            dtype=np.int32
        )


    count = min(
        count,
        len(coords)
    )


    indices = np.random.choice(
        len(coords),
        size=count,
        replace=False
    )


    return coords[
        indices
    ]


# ============================================================
# BUILD TRAINING SET
# ============================================================

print()
print("=" * 70)
print("BUILDING TRAINING SET")
print("=" * 70)


# ------------------------------------------------------------
# Positive samples
# ------------------------------------------------------------

positive_sample = sample_coords(
    positive_coords,
    POSITIVE_COUNT
)


# ------------------------------------------------------------
# Ordinary negatives
# ------------------------------------------------------------

ordinary_sample = sample_coords(
    ordinary_negative_coords,
    ORDINARY_NEGATIVE_COUNT
)


# ------------------------------------------------------------
# Very hard + hard
# ------------------------------------------------------------

hard_pool = np.concatenate(
    [
        very_hard,
        hard
    ],
    axis=0
)


hard_sample = sample_coords(
    hard_pool,
    HARD_NEGATIVE_COUNT
)


# ------------------------------------------------------------
# Moderate hard
# ------------------------------------------------------------

moderate_pool = np.concatenate(
    [
        moderate,
        easy_hard
    ],
    axis=0
)


moderate_sample = sample_coords(
    moderate_pool,
    MODERATE_HARD_COUNT
)


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

train_coords = np.concatenate(
    [
        positive_sample,
        ordinary_sample,
        hard_sample,
        moderate_sample
    ],
    axis=0
)


# ------------------------------------------------------------
# Shuffle
# ------------------------------------------------------------

shuffle_indices = np.random.permutation(
    len(train_coords)
)


train_coords = train_coords[
    shuffle_indices
]


print(
    "Positive samples:",
    len(
        positive_sample
    )
)


print(
    "Ordinary negatives:",
    len(
        ordinary_sample
    )
)


print(
    "Hard negatives:",
    len(
        hard_sample
    )
)


print(
    "Moderate hard negatives:",
    len(
        moderate_sample
    )
)


print(
    "TOTAL:",
    len(
        train_coords
    )
)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("CREATING DATASET")
print("=" * 70)


dataset = FaultPatchDataset(
    features=features,
    labels=labels,
    feature_mask=feature_mask,
    coordinates=train_coords,
    means=means,
    stds=stds
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


print(
    "Batches per epoch:",
    len(loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL V2")
print("=" * 70)


model = FaultSegmentationUNet(
    in_channels=19
).to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

pos_weight = torch.tensor(
    [4.0],
    dtype=torch.float32,
    device=DEVICE
)


bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ============================================================
# DISTANCE-AWARE TVERSKY LOSS
# ============================================================

def distance_tversky_loss(
    logits,
    targets
):

    probs = torch.sigmoid(
        logits
    )


    size = probs.shape[-1]

    center = size // 2


    yy, xx = torch.meshgrid(
        torch.arange(
            size,
            device=probs.device
        ),

        torch.arange(
            size,
            device=probs.device
        ),

        indexing="ij"
    )


    distance = torch.sqrt(
        (
            (
                yy - center
            ).float()
            ** 2
        )
        +
        (
            (
                xx - center
            ).float()
            ** 2
        )
    )


    weight = torch.clamp(
        1.0
        -
        distance / 3.0,
        min=0.0
    )


    weight = weight.unsqueeze(
        0
    ).unsqueeze(
        0
    )


    # --------------------------------------------------------
    # Weighted TP
    # --------------------------------------------------------

    tp = torch.sum(
        weight
        *
        probs
        *
        targets
    )


    # --------------------------------------------------------
    # Weighted FP
    # --------------------------------------------------------

    fp = torch.sum(
        weight
        *
        probs
        *
        (
            1.0
            -
            targets
        )
    )


    # --------------------------------------------------------
    # Weighted FN
    # --------------------------------------------------------

    fn = torch.sum(
        weight
        *
        (
            1.0
            -
            probs
        )
        *
        targets
    )


    alpha = 0.2
    beta = 0.8


    score = (
        tp
        /
        (
            tp
            +
            alpha * fp
            +
            beta * fn
            +
            1e-6
        )
    )


    return 1.0 - score


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING MODEL V2")
print("=" * 70)


best_loss = float(
    "inf"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    running_loss = 0.0

    running_bce = 0.0

    running_tv = 0.0

    batches = 0


    for batch_index, (
        patches,
        targets
    ) in enumerate(
        loader,
        start=1
    ):

        patches = patches.to(
            DEVICE
        )


        targets = targets.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        if not torch.isfinite(
            patches
        ).all():

            print(
                "\nWARNING: "
                "non-finite input detected."
            )

            raise RuntimeError(
                "Non-finite patch reached model."
            )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            patches
        )


        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        loss_bce = bce_loss(
            logits,
            targets
        )


        loss_tv = distance_tversky_loss(
            logits,
            targets
        )


        loss = (
            0.5 * loss_bce
            +
            0.5 * loss_tv
        )


        # ----------------------------------------------------
        # Backpropagation
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

        running_loss += (
            loss.item()
        )


        running_bce += (
            loss_bce.item()
        )


        running_tv += (
            loss_tv.item()
        )


        batches += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index % 250
            == 0
        ):

            print(
                f"\rEpoch {epoch}/{EPOCHS} "
                f"Batch {batch_index}/{len(loader)}",
                end="",
                flush=True
            )


    # ========================================================
    # EPOCH STATISTICS
    # ========================================================

    train_loss = (
        running_loss
        /
        batches
    )


    train_bce = (
        running_bce
        /
        batches
    )


    train_tv = (
        running_tv
        /
        batches
    )


    scheduler.step(
        train_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print()
    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )


    print(
        f"Train Loss: "
        f"{train_loss:.6f}"
    )


    print(
        f"Train BCE : "
        f"{train_bce:.6f}"
    )


    print(
        f"Train TV  : "
        f"{train_tv:.6f}"
    )


    print(
        f"LR        : "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if train_loss < best_loss:

        best_loss = train_loss


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_bce":
                    train_bce,

                "train_tversky":
                    train_tv
            },
            BEST_MODEL_PATH
        )


        print(
            "Saved BEST model."
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

        "train_loss":
            train_loss,

        "train_bce":
            train_bce,

        "train_tversky":
            train_tv
    },
    FINAL_MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)


print(
    "Best model:",
    BEST_MODEL_PATH
)


print(
    "Final model:",
    FINAL_MODEL_PATH
)


print()
print(
    "Validation blocks were NOT used for training."
)


print(
    "Next: evaluate Model V2 on the untouched "
    "validation raster."
)