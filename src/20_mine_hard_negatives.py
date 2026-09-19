import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

MODEL_PATH = "models/fault_tversky_unet_best.pt"

OUTPUT_DIR = "data/processed/hard_negatives"

OUTPUT_COORDS = os.path.join(
    OUTPUT_DIR,
    "hard_negative_coords.npy"
)

OUTPUT_SCORES = os.path.join(
    OUTPUT_DIR,
    "hard_negative_scores.npy"
)

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

# Pixels predicted above this probability are candidates.
HARD_NEGATIVE_THRESHOLD = 0.30

# Don't select pixels close to known faults.
MIN_DISTANCE_FROM_FAULT = 5

# Maximum hard negatives to save.
MAX_HARD_NEGATIVES = 100_000

# Inference tile size.
TILE_SIZE = 256


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
# U-NET BUILDING BLOCK
# EXACTLY MATCHES src/14_train_tversky_unet.py
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
# EXACTLY MATCHES src/14_train_tversky_unet.py
# ============================================================

class FaultSegmentationUNet(nn.Module):

    def __init__(self, in_channels=19):

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
        # Output
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
# LOAD NORMALIZATION
# ============================================================

def load_normalization():

    mean_path = (
        "data/processed/cnn/"
        "feature_means.npy"
    )

    std_path = (
        "data/processed/cnn/"
        "feature_stds.npy"
    )

    means = np.load(
        mean_path
    ).astype(
        np.float32
    )

    stds = np.load(
        std_path
    ).astype(
        np.float32
    )

    stds = np.maximum(
        stds,
        1e-6
    )

    return means, stds


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(
    features,
    valid_mask,
    means,
    stds
):

    features = features.astype(
        np.float32,
        copy=False
    )

    for b in range(
        features.shape[0]
    ):

        band = features[b]

        invalid = (
            ~valid_mask
            |
            ~np.isfinite(band)
        )

        # IMPORTANT:
        # Replace invalid values BEFORE normalization.
        band[invalid] = means[b]

        features[b] = (
            (band - means[b])
            / stds[b]
        )

    # Same clipping used during training.
    features = np.clip(
        features,
        -10.0,
        10.0
    )

    # Nodata pixels become zero.
    features[
        :,
        ~valid_mask
    ] = 0.0

    return features


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = FaultSegmentationUNet(
    in_channels=19
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Handle either a raw state_dict or
# a checkpoint containing model_state_dict.

if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.eval()

print(
    "Successfully loaded:",
    MODEL_PATH
)


# ============================================================
# LOAD RASTERS
# ============================================================

print()
print("=" * 70)
print("LOADING RASTERS")
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

    H = src.height
    W = src.width


print(
    "Features shape:",
    features.shape
)


with rasterio.open(
    LABELS_PATH
) as src:

    labels = src.read(1)


print(
    "Labels shape:",
    labels.shape
)


# ============================================================
# CREATE TRAINING MASK
# ============================================================

print()
print("=" * 70)
print("CREATING TRAINING MASK")
print("=" * 70)


rows_per_block = H // 4
cols_per_block = W // 4


# These are NEVER used for hard-negative mining.
validation_blocks = {
    1,
    7,
    9,
    10
}


training_mask = np.zeros(
    (H, W),
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
        else H
    )

    x0 = (
        block_col
        * cols_per_block
    )

    x1 = (
        (block_col + 1)
        * cols_per_block
        if block_col < 3
        else W
    )

    if block_id not in validation_blocks:

        training_mask[
            y0:y1,
            x0:x1
        ] = True


print(
    "Training pixels:",
    int(training_mask.sum())
)

print(
    "Validation pixels:",
    int((~training_mask).sum())
)


# ============================================================
# PATCH-SAFE TRAINING CENTERS
# ============================================================

print()
print("=" * 70)
print("CREATING SAFE TRAINING CENTERS")
print("=" * 70)


safe_training_mask = (
    training_mask.copy()
)


# A 31x31 patch requires 15 pixels
# of context on every side.

safe_training_mask[
    :RADIUS,
    :
] = False

safe_training_mask[
    -RADIUS:,
    :
] = False

safe_training_mask[
    :,
    :RADIUS
] = False

safe_training_mask[
    :,
    -RADIUS:
] = False


# Ensure the entire patch stays in
# the training region.

distance_from_validation = (
    distance_transform_edt(
        training_mask
    )
)


safe_training_mask &= (
    distance_from_validation
    > RADIUS
)


print(
    "Safe training centers:",
    int(
        safe_training_mask.sum()
    )
)


# ============================================================
# REMOVE KNOWN FAULT AREAS
# ============================================================

print()
print("=" * 70)
print("REMOVING KNOWN FAULT AREAS")
print("=" * 70)


fault_mask = (
    labels > 0
)


distance_from_fault = (
    distance_transform_edt(
        ~fault_mask
    )
)


far_from_fault = (
    distance_from_fault
    >= MIN_DISTANCE_FROM_FAULT
)


candidate_mask = (
    safe_training_mask
    &
    feature_mask
    &
    (labels == 0)
    &
    far_from_fault
)


print(
    "Candidate negative pixels:",
    int(
        candidate_mask.sum()
    )
)


# ============================================================
# NORMALIZATION
# ============================================================

means, stds = load_normalization()


# ============================================================
# FULL TRAINING-REGION INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING TRAINING-REGION INFERENCE")
print("=" * 70)


probability_map = np.zeros(
    (H, W),
    dtype=np.float32
)


count_map = np.zeros(
    (H, W),
    dtype=np.uint16
)


context = RADIUS


num_tiles_y = (
    H + TILE_SIZE - 1
) // TILE_SIZE


num_tiles_x = (
    W + TILE_SIZE - 1
) // TILE_SIZE


total_tiles = (
    num_tiles_y
    *
    num_tiles_x
)


tile_number = 0


with torch.no_grad():

    for y0 in range(
        0,
        H,
        TILE_SIZE
    ):

        for x0 in range(
            0,
            W,
            TILE_SIZE
        ):

            tile_number += 1

            print(
                f"\rTile "
                f"{tile_number}/"
                f"{total_tiles}",
                end="",
                flush=True
            )

            # ------------------------------------------------
            # Read tile with context.
            # ------------------------------------------------

            read_y0 = max(
                0,
                y0 - context
            )

            read_y1 = min(
                H,
                y0
                + TILE_SIZE
                + context
            )

            read_x0 = max(
                0,
                x0 - context
            )

            read_x1 = min(
                W,
                x0
                + TILE_SIZE
                + context
            )


            tile = features[
                :,
                read_y0:read_y1,
                read_x0:read_x1
            ].copy()


            tile_valid = (
                feature_mask[
                    read_y0:read_y1,
                    read_x0:read_x1
                ]
            )


            tile = normalize_features(
                tile,
                tile_valid,
                means,
                stds
            )


            tensor = (
                torch.from_numpy(tile)
                .unsqueeze(0)
                .to(DEVICE)
            )


            logits = model(
                tensor
            )


            probs = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()


            # ------------------------------------------------
            # Write central tile.
            # ------------------------------------------------

            write_y0 = y0

            write_y1 = min(
                H,
                y0 + TILE_SIZE
            )

            write_x0 = x0

            write_x1 = min(
                W,
                x0 + TILE_SIZE
            )


            local_y0 = (
                write_y0
                - read_y0
            )

            local_y1 = (
                write_y1
                - read_y0
            )

            local_x0 = (
                write_x0
                - read_x0
            )

            local_x1 = (
                write_x1
                - read_x0
            )


            probability_map[
                write_y0:write_y1,
                write_x0:write_x1
            ] += probs[
                local_y0:local_y1,
                local_x0:local_x1
            ]


            count_map[
                write_y0:write_y1,
                write_x0:write_x1
            ] += 1


print()


# ============================================================
# AVERAGE OVERLAPS
# ============================================================

print()
print("=" * 70)
print("AVERAGING PREDICTIONS")
print("=" * 70)


valid_counts = (
    count_map > 0
)


probability_map[
    valid_counts
] /= count_map[
    valid_counts
]


probability_map = np.clip(
    probability_map,
    0.0,
    1.0
)


# ============================================================
# EXTRACT HARD NEGATIVES
# ============================================================

print()
print("=" * 70)
print("EXTRACTING HARD NEGATIVES")
print("=" * 70)


hard_mask = (
    candidate_mask
    &
    (
        probability_map
        >= HARD_NEGATIVE_THRESHOLD
    )
)


hard_coords = np.argwhere(
    hard_mask
)


hard_scores = (
    probability_map[
        hard_mask
    ]
)


print(
    "Hard-negative candidates:",
    len(hard_coords)
)


# ============================================================
# LIMIT NUMBER
# ============================================================

if (
    len(hard_coords)
    > MAX_HARD_NEGATIVES
):

    print(
        "Limiting to top",
        MAX_HARD_NEGATIVES,
        "hard negatives."
    )

    order = np.argsort(
        hard_scores
    )[::-1]

    order = order[
        :MAX_HARD_NEGATIVES
    ]

    hard_coords = (
        hard_coords[order]
    )

    hard_scores = (
        hard_scores[order]
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.save(
    OUTPUT_COORDS,
    hard_coords.astype(
        np.int32
    )
)


np.save(
    OUTPUT_SCORES,
    hard_scores.astype(
        np.float32
    )
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("HARD-NEGATIVE RESULTS")
print("=" * 70)


print(
    "Coordinates:",
    hard_coords.shape
)

print(
    "Scores:",
    hard_scores.shape
)


if len(hard_scores) > 0:

    print(
        "Min score:",
        float(hard_scores.min())
    )

    print(
        "Max score:",
        float(hard_scores.max())
    )

    print(
        "Mean score:",
        float(hard_scores.mean())
    )

    print(
        "Median score:",
        float(np.median(hard_scores))
    )


    print()

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]:

        count = np.sum(
            hard_scores
            >= threshold
        )

        print(
            f">= {threshold:.2f}: "
            f"{count}"
        )

else:

    print(
        "No hard negatives found."
    )


print()
print("Saved:")
print(
    OUTPUT_COORDS
)
print(
    OUTPUT_SCORES
)

print()
print("Done.")