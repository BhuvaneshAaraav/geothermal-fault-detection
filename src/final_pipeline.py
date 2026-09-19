"""
GeoDAWN FINAL PRODUCTION PIPELINE

V1 Tversky U-Net
----------------

Input:
    19-band geophysical raster

Inference:
    31x31 spatial context
    256x256 tiles
    32-pixel overlap
    exact Script-15 inference behavior

Post-processing:
    power = 0.001
    spatial radius = 1 pixel (~100 m)
    K = 91,000

IMPORTANT:
    Validation champion:
        DTI = 0.239444373

    This score was obtained on the development validation set.
    It is NOT an unbiased estimate of performance on unseen data.

    Raw model probability and ranking score are different quantities.
    Candidate locations are model predictions, not geological confirmation.
"""


import argparse
import csv
import os

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL = (
    "models/fault_tversky_unet_best.pt"
)

DEFAULT_MEANS = (
    "data/processed/cnn/feature_means.npy"
)

DEFAULT_STDS = (
    "data/processed/cnn/feature_stds.npy"
)

DEFAULT_K = 91_000

DEFAULT_POWER = 0.001

DEFAULT_RADIUS = 1

TILE_SIZE = 256

OVERLAP = 32

PATCH_SIZE = 31

CONTEXT = PATCH_SIZE // 2


# ============================================================
# V1 MODEL
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


class FaultUNet(nn.Module):

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

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.enc3 = ConvBlock(
            64,
            128
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # ----------------------------------------------------
        # Pool
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            2
        )

        # ----------------------------------------------------
        # Learned upsampling
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.dec3 = ConvBlock(
            128 + 128,
            128
        )

        self.dec2 = ConvBlock(
            64 + 64,
            64
        )

        self.dec1 = ConvBlock(
            32 + 32,
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

        # ====================================================
        # Encoder
        # ====================================================

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck(
            self.pool(e3)
        )

        # ====================================================
        # Decoder 3
        # ====================================================

        d3 = self.up3(
            b
        )

        if d3.shape[-2:] != e3.shape[-2:]:

            d3 = F.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        # ====================================================
        # Decoder 2
        # ====================================================

        d2 = self.up2(
            d3
        )

        if d2.shape[-2:] != e2.shape[-2:]:

            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(
            d2
        )

        # ====================================================
        # Decoder 1
        # ====================================================

        d1 = self.up1(
            d2
        )

        if d1.shape[-2:] != e1.shape[-2:]:

            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(
            d1
        )

        return self.final(
            d1
        )


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    device
):

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(
        "Model:",
        model_path
    )

    model = FaultUNet(
        in_channels=19
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        if "val_tversky" in checkpoint:

            print(
                "Validation Tversky:",
                checkpoint[
                    "val_tversky"
                ]
            )

        if "val_roc" in checkpoint:

            print(
                "Validation ROC:",
                checkpoint[
                    "val_roc"
                ]
            )

        if "val_pr" in checkpoint:

            print(
                "Validation PR:",
                checkpoint[
                    "val_pr"
                ]
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(
        device
    )

    model.eval()

    print(
        "Device:",
        device
    )

    return model


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    input_path,
    means_path,
    stds_path
):

    print()
    print("=" * 70)
    print("LOADING FEATURES")
    print("=" * 70)

    with rasterio.open(
        input_path
    ) as src:

        data = src.read(
            out_dtype="float32"
        )

        feature_mask = (
            src.read_masks(1) > 0
        )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

    print(
        "Shape:",
        data.shape
    )

    print(
        "CRS:",
        crs
    )

    print(
        "Resolution:",
        transform.a,
        transform.e
    )

    means = np.load(
        means_path
    ).astype(
        np.float32
    )

    stds = np.load(
        stds_path
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

    if data.shape[0] != 19:

        raise ValueError(
            "Expected 19 input bands."
        )

    if means.shape[0] != 19:

        raise ValueError(
            "Expected 19 normalization means."
        )

    if stds.shape[0] != 19:

        raise ValueError(
            "Expected 19 normalization stds."
        )

    # ========================================================
    # NORMALIZATION
    #
    # Exact invalid-value handling used by verified
    # standalone inference.
    # ========================================================

    invalid_total = 0

    for band in range(
        19
    ):

        invalid = (
            (~feature_mask)
            |
            (~np.isfinite(
                data[band]
            ))
            |
            (
                np.abs(
                    data[band]
                ) > 1e30
            )
        )

        invalid_total += int(
            invalid.sum()
        )

        data[band][
            invalid
        ] = means[band]

        data[band] -= means[band]

        data[band] /= max(
            stds[band],
            1e-8
        )

    np.clip(
        data,
        -10.0,
        10.0,
        out=data
    )

    data[
        :,
        ~feature_mask
    ] = 0.0

    print(
        "Invalid values handled:",
        f"{invalid_total:,}"
    )

    print(
        "Normalized range:",
        float(data.min()),
        "to",
        float(data.max())
    )

    return (
        data,
        feature_mask,
        profile,
        transform,
        crs
    )


# ============================================================
# EXACT SCRIPT 15 INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    features,
    feature_mask,
    model,
    device
):

    bands, height, width = (
        features.shape
    )

    print()
    print("=" * 70)
    print("RUNNING FULL RASTER INFERENCE")
    print("=" * 70)

    print(
        "Raster:",
        height,
        "x",
        width
    )

    print(
        "Tile size:",
        TILE_SIZE
    )

    print(
        "Overlap:",
        OVERLAP
    )

    print(
        "Context:",
        CONTEXT
    )

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Exact Script 15 tile positions
    # --------------------------------------------------------

    step = (
        TILE_SIZE
        -
        OVERLAP
    )

    tile_y_positions = list(
        range(
            0,
            height,
            step
        )
    )

    tile_x_positions = list(
        range(
            0,
            width,
            step
        )
    )

    total_tiles = (
        len(tile_y_positions)
        *
        len(tile_x_positions)
    )

    tile_number = 0

    # ========================================================
    # TILES
    # ========================================================

    for y0 in tile_y_positions:

        for x0 in tile_x_positions:

            tile_number += 1

            # ------------------------------------------------
            # Core tile
            # ------------------------------------------------

            y1 = min(
                y0 + TILE_SIZE,
                height
            )

            x1 = min(
                x0 + TILE_SIZE,
                width
            )

            # ------------------------------------------------
            # Context
            #
            # IMPORTANT:
            # NO padding.
            # Exact Script 15 behavior.
            # ------------------------------------------------

            cy0 = max(
                0,
                y0 - CONTEXT
            )

            cx0 = max(
                0,
                x0 - CONTEXT
            )

            cy1 = min(
                height,
                y1 + CONTEXT
            )

            cx1 = min(
                width,
                x1 + CONTEXT
            )

            tile = features[
                :,
                cy0:cy1,
                cx0:cx1
            ]

            # ------------------------------------------------
            # Predict tile
            # ------------------------------------------------

            tensor = torch.from_numpy(
                tile
            ).unsqueeze(
                0
            ).to(
                device,
                dtype=torch.float32
            )

            logits = model(
                tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            pred = probabilities[
                0,
                0
            ].cpu().numpy()

            # ------------------------------------------------
            # Coordinates inside prediction
            # ------------------------------------------------

            py0 = (
                y0
                -
                cy0
            )

            px0 = (
                x0
                -
                cx0
            )

            py1 = (
                py0
                +
                (y1 - y0)
            )

            px1 = (
                px0
                +
                (x1 - x0)
            )

            core_prediction = pred[
                py0:py1,
                px0:px1
            ]

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            prediction_sum[
                y0:y1,
                x0:x1
            ] += core_prediction

            prediction_count[
                y0:y1,
                x0:x1
            ] += 1.0

            if (
                tile_number == 1
                or tile_number % 20 == 0
                or tile_number == total_tiles
            ):

                print(
                    f"\rTiles: "
                    f"{tile_number}/"
                    f"{total_tiles}",
                    end=""
                )

    print()

    # ========================================================
    # AVERAGE
    # ========================================================

    full_prediction = (
        prediction_sum
        /
        np.maximum(
            prediction_count,
            1.0
        )
    )

    # ========================================================
    # MASK
    # ========================================================

    full_prediction[
        ~feature_mask
    ] = 0.0

    # ========================================================
    # CLIP
    # ========================================================

    full_prediction = np.clip(
        full_prediction,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    print()
    print(
        "Inference complete."
    )

    print(
        "Prediction min:",
        float(
            full_prediction.min()
        )
    )

    print(
        "Prediction median:",
        float(
            np.median(
                full_prediction
            )
        )
    )

    print(
        "Prediction mean:",
        float(
            full_prediction.mean()
        )
    )

    print(
        "Prediction P90:",
        float(
            np.percentile(
                full_prediction,
                90
            )
        )
    )

    print(
        "Prediction P99:",
        float(
            np.percentile(
                full_prediction,
                99
            )
        )
    )

    print(
        "Prediction max:",
        float(
            full_prediction.max()
        )
    )

    return full_prediction


# ============================================================
# SAVE RAW PROBABILITY RASTER
# ============================================================

def save_probability_raster(
    prediction,
    input_path,
    output_path
):

    print()
    print("=" * 70)
    print("SAVING RAW PROBABILITY RASTER")
    print("=" * 70)

    with rasterio.open(
        input_path
    ) as src:

        profile = src.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            prediction.astype(
                np.float32
            ),
            1
        )

        dst.set_band_description(
            1,
            "Raw V1 U-Net fault probability"
        )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# EXACT SCRIPT 51 SPATIAL THINNING
# ============================================================

def spatial_thin(
    scores,
    coords,
    K,
    radius,
    height,
    width
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Sort ALL candidates.
    #
    # Do NOT perform Top-K before thinning.
    # --------------------------------------------------------

    order = np.argsort(
        scores
    )[::-1]

    selected = []

    occupied = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for idx in order:

        y, x = coords[idx]

        y = int(y)

        x = int(x)

        y0 = max(
            0,
            y - radius
        )

        y1 = min(
            height,
            y + radius + 1
        )

        x0 = max(
            0,
            x - radius
        )

        x1 = min(
            width,
            x + radius + 1
        )

        if occupied[
            y0:y1,
            x0:x1
        ].any():

            continue

        selected.append(
            idx
        )

        occupied[
            y0:y1,
            x0:x1
        ] = 1

        # ----------------------------------------------------
        # Stop ONLY after an accepted candidate.
        # ----------------------------------------------------

        if len(selected) >= K:

            break

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# FINAL CANDIDATE EXTRACTION
# ============================================================

def create_candidates(
    prediction,
    feature_mask,
    K,
    power,
    radius
):

    print()
    print("=" * 70)
    print("POST-PROCESSING")
    print("=" * 70)

    height, width = (
        prediction.shape
    )

    # --------------------------------------------------------
    # ALL VALID PIXELS
    # --------------------------------------------------------

    valid_flat = np.flatnonzero(
        feature_mask.ravel()
    )

    valid_scores = prediction.ravel()[
        valid_flat
    ]

    # --------------------------------------------------------
    # Remove zero predictions
    # --------------------------------------------------------

    positive = (
        valid_scores > 0.0
    )

    valid_flat = valid_flat[
        positive
    ]

    valid_scores = valid_scores[
        positive
    ]

    print(
        "Valid positive-score pixels:",
        f"{len(valid_flat):,}"
    )

    if len(valid_flat) == 0:

        raise RuntimeError(
            "No positive predictions found."
        )

    # --------------------------------------------------------
    # Convert flat indexes to coordinates
    # --------------------------------------------------------

    rows = (
        valid_flat // width
    )

    cols = (
        valid_flat % width
    )

    coords = np.column_stack(
        [
            rows,
            cols
        ]
    ).astype(
        np.int64
    )

    # --------------------------------------------------------
    # Power transformation
    #
    # IMPORTANT:
    # Apply to ALL valid pixels before thinning.
    # --------------------------------------------------------

    ranking_scores = np.power(
        np.clip(
            valid_scores,
            0.0,
            1.0
        ),
        power
    )

    # --------------------------------------------------------
    # EXACT SCRIPT 51 THINNING
    # --------------------------------------------------------

    selected_idx = spatial_thin(
        ranking_scores,
        coords,
        K,
        radius,
        height,
        width
    )

    # --------------------------------------------------------
    # Selected data
    # --------------------------------------------------------

    selected_flat = valid_flat[
        selected_idx
    ]

    selected_rows = rows[
        selected_idx
    ]

    selected_cols = cols[
        selected_idx
    ]

    selected_raw = valid_scores[
        selected_idx
    ]

    selected_ranking = ranking_scores[
        selected_idx
    ]

    print(
        "Requested K:",
        f"{K:,}"
    )

    print(
        "Selected:",
        f"{len(selected_idx):,}"
    )

    print(
        "Power:",
        power
    )

    print(
        "Spatial radius:",
        radius,
        "pixel"
    )

    if len(selected_raw) > 0:

        print(
            "Raw probability range:",
            float(
                selected_raw.min()
            ),
            "to",
            float(
                selected_raw.max()
            )
        )

        print(
            "Ranking score range:",
            float(
                selected_ranking.min()
            ),
            "to",
            float(
                selected_ranking.max()
            )
        )

    return (
        selected_rows,
        selected_cols,
        selected_raw,
        selected_ranking,
        selected_flat
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    output_path,
    rows,
    cols,
    raw_scores,
    ranking_scores,
    transform,
    crs
):

    print()
    print("=" * 70)
    print("SAVING CANDIDATE LOCATIONS")
    print("=" * 70)

    from pyproj import Transformer

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "rank",
                "row",
                "column",
                "raw_model_probability",
                "ranking_score",
                "x",
                "y",
                "longitude",
                "latitude"
            ]
        )

        for rank, (
            row,
            col,
            raw,
            score
        ) in enumerate(
            zip(
                rows,
                cols,
                raw_scores,
                ranking_scores
            ),
            start=1
        ):

            x, y = rasterio.transform.xy(
                transform,
                int(row),
                int(col),
                offset="center"
            )

            longitude, latitude = (
                transformer.transform(
                    x,
                    y
                )
            )

            writer.writerow(
                [
                    rank,
                    int(row),
                    int(col),
                    f"{raw:.9f}",
                    f"{score:.9f}",
                    f"{x:.3f}",
                    f"{y:.3f}",
                    f"{longitude:.8f}",
                    f"{latitude:.8f}"
                ]
            )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SAVE MAP
# ============================================================

def save_candidate_map(
    prediction,
    feature_mask,
    rows,
    cols,
    output_path
):

    print()
    print("=" * 70)
    print("SAVING CANDIDATE MAP")
    print("=" * 70)

    import matplotlib.pyplot as plt

    display = np.log10(
        np.maximum(
            prediction,
            1e-6
        )
    )

    display[
        ~feature_mask
    ] = np.nan

    plt.figure(
        figsize=(12, 9)
    )

    plt.imshow(
        display,
        origin="upper"
    )

    plt.scatter(
        cols,
        rows,
        s=2
    )

    plt.title(
        "GeoDAWN - V1 Fault Candidate Map"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.colorbar(
        label=(
            "log10("
            "raw model probability"
            ")"
        )
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_config(
    output_path,
    args,
    selected_count
):

    with open(
        output_path,
        "w"
    ) as f:

        f.write(
            "GeoDAWN Final Pipeline\n"
        )

        f.write(
            "======================\n\n"
        )

        f.write(
            f"Model: {args.model}\n"
        )

        f.write(
            "Architecture: V1 Tversky U-Net\n"
        )

        f.write(
            "Input bands: 19\n"
        )

        f.write(
            "Patch size: 31x31\n"
        )

        f.write(
            f"Tile size: {TILE_SIZE}\n"
        )

        f.write(
            f"Overlap: {OVERLAP}\n"
        )

        f.write(
            f"K: {args.k}\n"
        )

        f.write(
            f"Power: {args.power}\n"
        )

        f.write(
            f"Spatial radius: {args.radius} pixel\n"
        )

        f.write(
            f"Final candidates: {selected_count}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Validation champion DTI: "
            "0.239444373\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Important: candidate locations "
            "are model predictions, not "
            "geological confirmation.\n"
        )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "GeoDAWN V1 final "
            "fault-candidate inference pipeline"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "19-band input GeoTIFF"
        )
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/final",
        help=(
            "Output directory"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--means",
        default=DEFAULT_MEANS
    )

    parser.add_argument(
        "--stds",
        default=DEFAULT_STDS
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K
    )

    parser.add_argument(
        "--power",
        type=float,
        default=DEFAULT_POWER
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_RADIUS
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if args.k <= 0:

        raise ValueError(
            "--k must be greater than 0"
        )

    if args.power <= 0:

        raise ValueError(
            "--power must be greater than 0"
        )

    if args.radius < 0:

        raise ValueError(
            "--radius must be >= 0"
        )

    if not os.path.exists(
        args.input
    ):

        raise FileNotFoundError(
            args.input
        )

    if not os.path.exists(
        args.model
    ):

        raise FileNotFoundError(
            args.model
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    raw_output = os.path.join(
        args.output_dir,
        "raw_fault_probability.tif"
    )

    csv_output = os.path.join(
        args.output_dir,
        "fault_candidates.csv"
    )

    map_output = os.path.join(
        args.output_dir,
        "fault_candidate_map.png"
    )

    config_output = os.path.join(
        args.output_dir,
        "pipeline_config.txt"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GEODAWN FINAL PRODUCTION PIPELINE")
    print("=" * 70)

    print(
        "Device:",
        get_device()
    )

    print(
        "K:",
        f"{args.k:,}"
    )

    print(
        "Power:",
        args.power
    )

    print(
        "Spatial radius:",
        args.radius,
        "pixel(s)"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    device = get_device()

    model = load_model(
        args.model,
        device
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    (
        features,
        feature_mask,
        profile,
        transform,
        crs
    ) = load_features(
        args.input,
        args.means,
        args.stds
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    prediction = run_inference(
        features,
        feature_mask,
        model,
        device
    )

    # --------------------------------------------------------
    # Save raw probability
    # --------------------------------------------------------

    save_probability_raster(
        prediction,
        args.input,
        raw_output
    )

    # --------------------------------------------------------
    # Candidate extraction
    # --------------------------------------------------------

    (
        rows,
        cols,
        raw_scores,
        ranking_scores,
        selected_flat
    ) = create_candidates(
        prediction,
        feature_mask,
        args.k,
        args.power,
        args.radius
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_csv(
        csv_output,
        rows,
        cols,
        raw_scores,
        ranking_scores,
        transform,
        crs
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    save_candidate_map(
        prediction,
        feature_mask,
        rows,
        cols,
        map_output
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    save_config(
        config_output,
        args,
        len(rows)
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print("OUTPUTS")
    print("--------------------------------------")

    print(
        "Raw probability:",
        raw_output
    )

    print(
        "Candidate CSV:",
        csv_output
    )

    print(
        "Candidate map:",
        map_output
    )

    print(
        "Configuration:",
        config_output
    )

    print()
    print("FINAL CONFIGURATION")
    print("--------------------------------------")

    print(
        "Model: V1 Tversky U-Net"
    )

    print(
        "Input bands: 19"
    )

    print(
        "Context: 31 x 31"
    )

    print(
        "Tile: 256 x 256"
    )

    print(
        "Overlap:",
        OVERLAP
    )

    print(
        "K:",
        f"{args.k:,}"
    )

    print(
        "Power:",
        args.power
    )

    print(
        "Spatial thinning:",
        args.radius,
        "pixel (~100 m)"
    )

    print(
        "Final candidates:",
        f"{len(rows):,}"
    )

    print()
    print(
        "VALIDATED DEVELOPMENT RESULT"
    )

    print(
        "Exact DTI:",
        "0.239444373"
    )

    print()
    print(
        "IMPORTANT"
    )

    print(
        "Raw model probability = neural-network output."
    )

    print(
        "Ranking score = probability^power."
    )

    print(
        "Candidate locations = model predictions,"
        " not geological confirmation."
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()