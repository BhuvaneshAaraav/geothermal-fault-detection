import os
import numpy as np
import rasterio
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

MODEL_PATH = (
    "models/fault_hard_negative_unet_best.pt"
)

MEAN_PATH = (
    "data/processed/cnn/feature_means.npy"
)

STD_PATH = (
    "data/processed/cnn/feature_stds.npy"
)

OUTPUT_DIR = "outputs"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "hard_negative_validation_probability.npy"
)

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

TILE_SIZE = 256

# Same validation blocks used throughout.
VALIDATION_BLOCKS = {
    1,
    7,
    9,
    10
}

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# PRINT CONFIG
# ============================================================

print("=" * 70)
print("MODEL V2 VALIDATION EVALUATION")
print("=" * 70)

print(
    "Using device:",
    DEVICE
)


# ============================================================
# CONV BLOCK
# EXACT MODEL 14 ARCHITECTURE
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
# U-NET
# EXACT MODEL 14 ARCHITECTURE
# ============================================================

class FaultSegmentationUNet(
    nn.Module
):

    def __init__(
        self,
        in_channels=19
    ):

        super().__init__()

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

        self.bottleneck = ConvBlock(
            128,
            256
        )

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
    "Loaded:",
    MODEL_PATH
)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

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

stds = np.maximum(
    stds,
    1e-6
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
# BUILD VALIDATION MASK
# ============================================================

print()
print("=" * 70)
print("BUILDING VALIDATION MASK")
print("=" * 70)

rows_per_block = height // 4
cols_per_block = width // 4

validation_mask = np.zeros(
    (height, width),
    dtype=bool
)


for block_id in VALIDATION_BLOCKS:

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

    validation_mask[
        y0:y1,
        x0:x1
    ] = True


# Only evaluate valid raster pixels.

evaluation_mask = (
    validation_mask
    &
    feature_mask
)


print(
    "Validation pixels:",
    int(
        validation_mask.sum()
    )
)

print(
    "Valid evaluation pixels:",
    int(
        evaluation_mask.sum()
    )
)

print(
    "Validation fault pixels:",
    int(
        (
            labels
            &
            validation_mask
        ).sum()
    )
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_tile(
    tile,
    valid_mask
):

    tile = tile.astype(
        np.float32,
        copy=False
    )

    for band in range(
        tile.shape[0]
    ):

        values = tile[
            band
        ]

        invalid = (
            ~valid_mask
            |
            ~np.isfinite(values)
            |
            (np.abs(values) > 1e30)
        )

        values[invalid] = (
            means[band]
        )

        tile[
            band
        ] = values


    tile = (
        tile
        -
        means[
            :,
            None,
            None
        ]
    ) / (
        stds[
            :,
            None,
            None
        ]
    )


    tile = np.clip(
        tile,
        -10.0,
        10.0
    )


    tile[
        :,
        ~valid_mask
    ] = 0.0


    tile[
        ~np.isfinite(tile)
    ] = 0.0


    return tile.astype(
        np.float32
    )


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("RUNNING VALIDATION INFERENCE")
print("=" * 70)

probability_map = np.zeros(
    (height, width),
    dtype=np.float32
)

count_map = np.zeros(
    (height, width),
    dtype=np.uint16
)


# ------------------------------------------------------------
# Important:
#
# We only need validation blocks, but context can extend
# outside validation blocks. This is okay because those
# pixels are only used as CNN spatial context.
#
# We never score them.
# ------------------------------------------------------------

tiles_y = range(
    0,
    height,
    TILE_SIZE
)

tiles_x = range(
    0,
    width,
    TILE_SIZE
)


num_tiles_y = (
    height
    + TILE_SIZE
    - 1
) // TILE_SIZE


num_tiles_x = (
    width
    + TILE_SIZE
    - 1
) // TILE_SIZE


total_tiles = (
    num_tiles_y
    *
    num_tiles_x
)


tile_number = 0


with torch.no_grad():

    for y0 in tiles_y:

        for x0 in tiles_x:

            tile_number += 1

            print(
                f"\rTile "
                f"{tile_number}/"
                f"{total_tiles}",
                end="",
                flush=True
            )


            # ------------------------------------------------
            # Context
            # ------------------------------------------------

            read_y0 = max(
                0,
                y0 - RADIUS
            )

            read_y1 = min(
                height,
                y0
                + TILE_SIZE
                + RADIUS
            )

            read_x0 = max(
                0,
                x0 - RADIUS
            )

            read_x1 = min(
                width,
                x0
                + TILE_SIZE
                + RADIUS
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


            tile = normalize_tile(
                tile,
                tile_valid
            )


            tensor = (
                torch.from_numpy(
                    tile
                )
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
            # Central tile
            # ------------------------------------------------

            write_y0 = y0

            write_y1 = min(
                height,
                y0 + TILE_SIZE
            )

            write_x0 = x0

            write_x1 = min(
                width,
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
# AVERAGE TILE PREDICTIONS
# ============================================================

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


# Only keep validation region.

validation_probability = np.zeros(
    (height, width),
    dtype=np.float32
)


validation_probability[
    evaluation_mask
] = probability_map[
    evaluation_mask
]


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.save(
    OUTPUT_PATH,
    validation_probability
)


# ============================================================
# COMPETITION METRIC
# ============================================================

def fast_metric(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )


    H, W = y_true.shape


    # --------------------------------------------------------
    # Build local offsets
    # --------------------------------------------------------

    offsets = []

    max_offset = int(
        np.ceil(RADIUS)
    )


    for dy in range(
        -max_offset,
        max_offset + 1
    ):

        for dx in range(
            -max_offset,
            max_offset + 1
        ):

            distance = np.sqrt(
                dy * dy
                +
                dx * dx
            )


            if distance <= RADIUS:

                kernel = max(
                    1.0
                    -
                    distance / RADIUS,
                    0.0
                )

                offsets.append(
                    (
                        dy,
                        dx,
                        kernel
                    )
                )


    # --------------------------------------------------------
    # GT matching
    # --------------------------------------------------------

    best_match = np.zeros_like(
        y_pred
    )


    for dy, dx, kernel in offsets:

        shifted = np.zeros_like(
            y_pred
        )


        src_y0 = max(
            0,
            -dy
        )

        src_y1 = min(
            H,
            H - dy
        )

        src_x0 = max(
            0,
            -dx
        )

        src_x1 = min(
            W,
            W - dx
        )


        dst_y0 = max(
            0,
            dy
        )

        dst_y1 = min(
            H,
            H + dy
        )

        dst_x0 = max(
            0,
            dx
        )

        dst_x1 = min(
            W,
            W + dx
        )


        shifted[
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = (
            y_pred[
                src_y0:src_y1,
                src_x0:src_x1
            ]
            *
            kernel
        )


        best_match = np.maximum(
            best_match,
            shifted
        )


    tp_w = np.sum(
        y_true
        *
        best_match
    )


    fn_w = np.sum(
        y_true
        *
        (
            1.0
            -
            best_match
        )
    )


    # --------------------------------------------------------
    # GT proximity for FP
    # --------------------------------------------------------

    gt_match = np.zeros_like(
        y_true
    )


    for dy, dx, kernel in offsets:

        shifted = np.zeros_like(
            y_true
        )


        src_y0 = max(
            0,
            -dy
        )

        src_y1 = min(
            H,
            H - dy
        )

        src_x0 = max(
            0,
            -dx
        )

        src_x1 = min(
            W,
            W - dx
        )


        dst_y0 = max(
            0,
            dy
        )

        dst_y1 = min(
            H,
            H + dy
        )

        dst_x0 = max(
            0,
            dx
        )

        dst_x1 = min(
            W,
            W + dx
        )


        shifted[
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = (
            y_true[
                src_y0:src_y1,
                src_x0:src_x1
            ]
            *
            kernel
        )


        gt_match = np.maximum(
            gt_match,
            shifted
        )


    fp_w = np.sum(
        y_pred
        *
        (
            1.0
            -
            gt_match
        )
    )


    # --------------------------------------------------------
    # Tversky
    # --------------------------------------------------------

    alpha = 0.2
    beta = 0.8


    denominator = (
        tp_w
        +
        alpha * fp_w
        +
        beta * fn_w
        +
        1e-12
    )


    score = (
        tp_w
        /
        denominator
    )


    return (
        score,
        tp_w,
        fp_w,
        fn_w
    )


# ============================================================
# EVALUATE
# ============================================================

print()
print("=" * 70)
print("EVALUATING COMPETITION METRIC")
print("=" * 70)


y_true = (
    labels
    *
    evaluation_mask
).astype(
    np.float64
)


y_pred = (
    validation_probability
    *
    evaluation_mask
).astype(
    np.float64
)


score, tp_w, fp_w, fn_w = fast_metric(
    y_true,
    y_pred
)


# ============================================================
# STATISTICS
# ============================================================

evaluation_values = (
    y_pred[
        evaluation_mask
    ]
)


print()
print("=" * 70)
print("MODEL V2 VALIDATION RESULTS")
print("=" * 70)


print(
    "Validation pixels:",
    int(
        evaluation_mask.sum()
    )
)


print(
    "Validation fault pixels:",
    int(
        y_true.sum()
    )
)


print(
    "Prediction min:",
    float(
        evaluation_values.min()
    )
)


print(
    "Prediction max:",
    float(
        evaluation_values.max()
    )
)


print(
    "Prediction mean:",
    float(
        evaluation_values.mean()
    )
)


print(
    "Prediction median:",
    float(
        np.median(
            evaluation_values
        )
    )
)


print()


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
        evaluation_values
        >= threshold
    )

    print(
        f"Prediction >= "
        f"{threshold:.1f}: "
        f"{count}"
    )


print()
print(
    "Weighted TP:",
    tp_w
)


print(
    "Weighted FP:",
    fp_w
)


print(
    "Weighted FN:",
    fn_w
)


print()
print(
    "Competition DTI:",
    score
)


# ============================================================
# BASELINES
# ============================================================

zero_prediction = np.zeros_like(
    y_true
)


zero_score, _, _, _ = fast_metric(
    y_true,
    zero_prediction
)


print()
print(
    "Zero prediction DTI:",
    zero_score
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)


print(
    "Previous V1 raw:        0.146870"
)


print(
    "Previous V1 calibrated: 0.174123"
)


print(
    f"Current V2 raw:        "
    f"{score:.6f}"
)


if score > 0.174123:

    print()
    print(
        "V2 raw score is above the "
        "previous calibrated baseline."
    )

elif score > 0.146870:

    print()
    print(
        "V2 improved over V1 raw, "
        "but is below the previous "
        "calibrated result."
    )

else:

    print()
    print(
        "V2 did not improve over "
        "the previous V1 raw result."
    )


print()
print(
    "Saved probability map:"
)

print(
    OUTPUT_PATH
)

print()
print("Done.")