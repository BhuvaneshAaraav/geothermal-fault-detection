import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_PATH = "models/fault_v3_unet_best.pt"

OUTPUT_DIR = "outputs"

PROBABILITY_PATH = os.path.join(
    OUTPUT_DIR,
    "v3_validation_probability.npy"
)

PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    "v3_validation_predictions.npy"
)

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

TILE_SIZE = 256

ALPHA = 0.2
BETA = 0.8
DISTANCE_RADIUS = 3

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

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
# MODEL
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

    def __init__(self):

        super().__init__()

        self.enc1 = ConvBlock(19, 32)
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
# SPATIAL VALIDATION MASK
# ============================================================

def build_validation_mask(
    height,
    width
):

    mask = np.zeros(
        (height, width),
        dtype=bool
    )

    block_h = height // 4
    block_w = width // 4

    for block_id in VAL_BLOCKS:

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

        mask[
            row_start:row_end,
            col_start:col_end
        ] = True

    return mask


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def normalize_features(
    values,
    means,
    stds
):

    values = values.astype(
        np.float32,
        copy=False
    )

    invalid = (
        ~np.isfinite(values)
        |
        (np.abs(values) > 1e30)
    )

    # Replace invalid values with band means

    for band in range(
        values.shape[0]
    ):

        band_invalid = invalid[band]

        if np.any(band_invalid):

            values[band][band_invalid] = means[band]

    values = (
        values
        -
        means[:, None, None]
    ) / (
        stds[:, None, None]
        +
        1e-8
    )

    values = np.clip(
        values,
        -10.0,
        10.0
    )

    invalid_any = np.any(
        invalid,
        axis=0
    )

    values[:, invalid_any] = 0.0

    return values


# ============================================================
# COMPETITION DTI
# ============================================================

def competition_dti(
    probability_map,
    labels,
    evaluation_mask,
    alpha=0.2,
    beta=0.8,
    radius=3
):

    print()
    print("=" * 70)
    print("COMPETITION DTI")
    print("=" * 70)

    # Ground truth only inside validation region

    gt = (
        labels > 0
    ) & evaluation_mask

    # Predictions only inside validation region

    probs = np.where(
        evaluation_mask,
        probability_map,
        0.0
    )

    # --------------------------------------------------------
    # Distance to nearest GT fault
    # --------------------------------------------------------

    distance = distance_transform_edt(
        ~gt
    )

    # --------------------------------------------------------
    # Weight based on distance
    #
    # distance 0 -> weight 1
    # distance 1 -> 1/2
    # distance 2 -> 1/3
    # distance 3 -> 1/4
    # farther   -> 0
    # --------------------------------------------------------

    weights = np.zeros_like(
        probs,
        dtype=np.float32
    )

    nearby = (
        distance <= radius
    )

    weights[nearby] = (
        1.0
        /
        (1.0 + distance[nearby])
    )

    # --------------------------------------------------------
    # Weighted true positives
    # --------------------------------------------------------

    weighted_tp = np.sum(
        probs
        *
        gt
        *
        weights
    )

    # --------------------------------------------------------
    # False positives
    #
    # A prediction on background is penalized,
    # but predictions close to faults receive
    # partial credit through the distance weighting.
    # --------------------------------------------------------

    weighted_fp = np.sum(
        probs
        *
        (~gt)
        *
        (
            1.0 - weights
        )
    )

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    weighted_fn = np.sum(
        (1.0 - probs)
        *
        gt
    )

    denominator = (
        weighted_tp
        +
        alpha * weighted_fp
        +
        beta * weighted_fn
        +
        1e-8
    )

    score = (
        weighted_tp
        /
        denominator
    )

    return (
        float(score),
        float(weighted_tp),
        float(weighted_fp),
        float(weighted_fn)
    )


# ============================================================
# TILED FULL-RASTER INFERENCE
# ============================================================

@torch.no_grad()
def run_full_inference(
    model,
    features,
    validation_mask,
    means,
    stds,
    height,
    width
):

    print()
    print("=" * 70)
    print("FULL VALIDATION INFERENCE")
    print("=" * 70)

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    model.eval()

    # --------------------------------------------------------
    # Process in overlapping tiles
    # --------------------------------------------------------

    total_tiles = (
        (
            height + TILE_SIZE - 1
        )
        // TILE_SIZE
    ) * (
        (
            width + TILE_SIZE - 1
        )
        // TILE_SIZE
    )

    tile_number = 0

    for row_start in range(
        0,
        height,
        TILE_SIZE
    ):

        for col_start in range(
            0,
            width,
            TILE_SIZE
        ):

            row_end = min(
                row_start + TILE_SIZE,
                height
            )

            col_end = min(
                col_start + TILE_SIZE,
                width
            )

            tile_number += 1

            # ------------------------------------------------
            # Expand tile so model has context
            # ------------------------------------------------

            read_row_start = max(
                0,
                row_start - RADIUS
            )

            read_col_start = max(
                0,
                col_start - RADIUS
            )

            read_row_end = min(
                height,
                row_end + RADIUS
            )

            read_col_end = min(
                width,
                col_end + RADIUS
            )

            tile = features[
                :,
                read_row_start:read_row_end,
                read_col_start:read_col_end
            ]

            # ------------------------------------------------
            # Pad tile if it is at raster boundary
            # ------------------------------------------------

            tile_h = tile.shape[1]
            tile_w = tile.shape[2]

            target_h = (
                row_end
                -
                row_start
                +
                2 * RADIUS
            )

            target_w = (
                col_end
                -
                col_start
                +
                2 * RADIUS
            )

            pad_bottom = max(
                0,
                target_h - tile_h
            )

            pad_right = max(
                0,
                target_w - tile_w
            )

            pad_top = max(
                0,
                RADIUS
                -
                (
                    row_start
                    -
                    read_row_start
                )
            )

            pad_left = max(
                0,
                RADIUS
                -
                (
                    col_start
                    -
                    read_col_start
                )
            )

            if (
                pad_top
                or pad_bottom
                or pad_left
                or pad_right
            ):

                tile = np.pad(
                    tile,
                    (
                        (0, 0),
                        (pad_top, pad_bottom),
                        (pad_left, pad_right)
                    ),
                    mode="edge"
                )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            tile = normalize_features(
                tile,
                means,
                stds
            )

            x = torch.from_numpy(
                tile
            ).unsqueeze(
                0
            ).to(
                DEVICE
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            output = model(x)

            output = torch.sigmoid(
                output
            )

            output = output[
                0,
                0
            ].cpu().numpy()

            # ------------------------------------------------
            # Extract requested tile
            # ------------------------------------------------

            prediction = output[
                RADIUS:
                RADIUS
                +
                (
                    row_end
                    -
                    row_start
                ),
                RADIUS:
                RADIUS
                +
                (
                    col_end
                    -
                    col_start
                )
            ]

            # ------------------------------------------------
            # Save only validation pixels
            # ------------------------------------------------

            tile_validation = validation_mask[
                row_start:row_end,
                col_start:col_end
            ]

            probability_map[
                row_start:row_end,
                col_start:col_end
            ][tile_validation] = prediction[
                tile_validation
            ]

            if (
                tile_number % 10 == 0
                or
                tile_number == total_tiles
            ):

                print(
                    f"Tiles: "
                    f"{tile_number}/{total_tiles}"
                )

    return probability_map


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V3 FULL VALIDATION EVALUATION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD NORMALIZATION
    # ========================================================

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

    # ========================================================
    # LOAD FEATURES
    # ========================================================

    print()
    print(
        "Loading feature raster..."
    )

    with rasterio.open(
        FEATURE_PATH
    ) as src:

        features = src.read().astype(
            np.float32
        )

        height = src.height
        width = src.width

    print(
        "Features:",
        features.shape
    )

    # ========================================================
    # LOAD LABELS
    # ========================================================

    print(
        "Loading labels..."
    )

    with rasterio.open(
        LABEL_PATH
    ) as src:

        labels = src.read(
            1
        )

    print(
        "Labels:",
        labels.shape
    )

    # ========================================================
    # VALIDATION MASK
    # ========================================================

    validation_mask = build_validation_mask(
        height,
        width
    )

    print()
    print(
        "Validation pixels:",
        int(
            validation_mask.sum()
        )
    )

    validation_faults = (
        (labels > 0)
        &
        validation_mask
    )

    print(
        "Validation fault pixels:",
        int(
            validation_faults.sum()
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading V3 best checkpoint..."
    )

    model = FaultSegmentationUNet().to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support both checkpoint formats

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        print(
            "Checkpoint PR-AUC:",
            checkpoint.get(
                "val_pr",
                checkpoint.get(
                    "best_val_pr",
                    "unknown"
                )
            )
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # FULL INFERENCE
    # ========================================================

    probability_map = run_full_inference(
        model,
        features,
        validation_mask,
        means,
        stds,
        height,
        width
    )

    # ========================================================
    # SAVE PROBABILITY MAP
    # ========================================================

    np.save(
        PROBABILITY_PATH,
        probability_map
    )

    print()
    print(
        "Saved:",
        PROBABILITY_PATH
    )

    # ========================================================
    # PROBABILITY STATISTICS
    # ========================================================

    valid_probs = probability_map[
        validation_mask
    ]

    print()
    print("=" * 70)
    print("PROBABILITY STATISTICS")
    print("=" * 70)

    print(
        "Min:",
        float(valid_probs.min())
    )

    print(
        "Max:",
        float(valid_probs.max())
    )

    print(
        "Mean:",
        float(valid_probs.mean())
    )

    print(
        "Median:",
        float(np.median(valid_probs))
    )

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
            valid_probs >= threshold
        )

        print(
            f">={threshold:.1f}:",
            int(count)
        )

    # ========================================================
    # COMPETITION DTI
    # ========================================================

    score, weighted_tp, weighted_fp, weighted_fn = \
        competition_dti(
            probability_map,
            labels,
            validation_mask,
            alpha=ALPHA,
            beta=BETA,
            radius=DISTANCE_RADIUS
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"V3 COMPETITION DTI: {score:.6f}"
    )

    print(
        f"Weighted TP: {weighted_tp:.3f}"
    )

    print(
        f"Weighted FP: {weighted_fp:.3f}"
    )

    print(
        f"Weighted FN: {weighted_fn:.3f}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # THRESHOLDED MAP
    # ========================================================

    # Keep raw probability map as primary output.
    # This thresholded map is only for inspection.

    threshold = 0.4

    prediction = (
        probability_map >= threshold
    ).astype(
        np.uint8
    )

    prediction[
        ~validation_mask
    ] = 0

    np.save(
        PREDICTION_PATH,
        prediction
    )

    print()
    print(
        "Saved:",
        PREDICTION_PATH
    )

    print(
        f"Threshold {threshold}:",
        int(
            prediction.sum()
        ),
        "predicted pixels"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("CURRENT MODEL COMPARISON")
    print("=" * 70)

    print(
        "V1 raw DTI       : 0.146870"
    )

    print(
        "V1 calibrated DTI: 0.180297*"
    )

    print(
        "V2 hard-negative : 0.109588"
    )

    print(
        f"V3 raw DTI       : {score:.6f}"
    )

    print()
    print(
        "* V1 0.180297 was obtained after "
        "tuning on this validation raster."
    )

    print(
        "For an honest model comparison, "
        "compare the raw DTI values first."
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()