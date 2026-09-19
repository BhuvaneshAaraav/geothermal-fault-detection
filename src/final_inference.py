import os
import argparse
import numpy as np
import rasterio
import torch
import torch.nn as nn


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_MODEL = "models/fault_tversky_unet_best.pt"

DEFAULT_MEANS = (
    "data/processed/cnn/feature_means.npy"
)

DEFAULT_STDS = (
    "data/processed/cnn/feature_stds.npy"
)

DEFAULT_OUTPUT = (
    "outputs/inference_fault_probability.tif"
)

TILE_SIZE = 256
OVERLAP = 32

BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")
    print("Using Apple MPS GPU")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")

else:

    DEVICE = torch.device("cpu")
    print("Using CPU")


# ============================================================
# MODEL
# EXACT V1 ARCHITECTURE
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

    def __init__(
        self,
        in_channels=19
    ):

        super().__init__()

        # Encoder
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
# LOAD MODEL
# ============================================================

def load_model(model_path):

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = FaultSegmentationUNet(
        in_channels=19
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Checkpoint format: state_dict"
        )

        print(
            "Training epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        print(
            "Validation Tversky:",
            checkpoint.get(
                "val_tversky",
                "unknown"
            )
        )

        print(
            "Validation ROC-AUC:",
            checkpoint.get(
                "val_roc_auc",
                "unknown"
            )
        )

        print(
            "Validation PR-AUC:",
            checkpoint.get(
                "val_pr_auc",
                "unknown"
            )
        )

    else:

        raise ValueError(
            "Unexpected checkpoint format."
        )

    model.eval()

    return model


# ============================================================
# LOAD + NORMALIZE RASTER
# ============================================================

def load_features(
    raster_path,
    means_path,
    stds_path
):

    print()
    print("=" * 70)
    print("LOADING INPUT RASTER")
    print("=" * 70)

    with rasterio.open(
        raster_path
    ) as src:

        features = src.read(
            out_dtype="float32"
        )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        raster_mask = (
            src.read_masks(1) > 0
        )

    print(
        "Input shape:",
        features.shape
    )

    print(
        "CRS:",
        crs
    )

    print(
        "Resolution:",
        transform.a,
        "x",
        abs(transform.e)
    )

    # --------------------------------------------------------
    # Validate number of bands
    # --------------------------------------------------------

    if features.shape[0] != 19:

        raise ValueError(
            f"Expected 19 bands, "
            f"got {features.shape[0]}"
        )

    # --------------------------------------------------------
    # Load normalization
    # --------------------------------------------------------

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

    if len(means) != 19:

        raise ValueError(
            "Means file does not contain "
            "19 values."
        )

    if len(stds) != 19:

        raise ValueError(
            "Stds file does not contain "
            "19 values."
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print()
    print("NORMALIZING FEATURES")

    total_invalid = 0

    for band in range(19):

        data = features[band]

        # IMPORTANT:
        # - non-finite values
        # - raster nodata mask
        # - extreme floating-point nodata values

        invalid = (
            (~raster_mask)
            |
            (~np.isfinite(data))
            |
            (np.abs(data) > 1e30)
        )

        count = int(
            invalid.sum()
        )

        total_invalid += count

        # Replace invalid values
        # with training mean

        data[invalid] = means[band]

        data -= means[band]

        data /= max(
            stds[band],
            1e-8
        )

        np.clip(
            data,
            -10.0,
            10.0,
            out=data
        )

    # All invalid pixels become zero
    # after normalization.

    features[
        :,
        ~raster_mask
    ] = 0.0

    features = np.ascontiguousarray(
        features,
        dtype=np.float32
    )

    print(
        "Invalid values handled:",
        f"{total_invalid:,}"
    )

    print(
        "Normalization complete."
    )

    return (
        features,
        profile,
        transform,
        crs,
        raster_mask
    )


# ============================================================
# TILE PREDICTION
# ============================================================

@torch.no_grad()
def predict_tile(
    model,
    tile
):

    tensor = torch.from_numpy(
        tile
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float32
    )

    logits = model(
        tensor
    )

    probabilities = torch.sigmoid(
        logits
    )

    return probabilities[
        0,
        0
    ].cpu().numpy()


# ============================================================
# FULL RASTER INFERENCE
# ============================================================

def run_inference(
    model,
    features
):

    height = features.shape[1]
    width = features.shape[2]

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    context = 15

    step = TILE_SIZE - OVERLAP

    y_positions = list(
        range(
            0,
            height,
            step
        )
    )

    x_positions = list(
        range(
            0,
            width,
            step
        )
    )

    total_tiles = (
        len(y_positions)
        * len(x_positions)
    )

    tile_number = 0

    print()
    print("=" * 70)
    print("RUNNING FULL-RASTER INFERENCE")
    print("=" * 70)

    for y0 in y_positions:

        for x0 in x_positions:

            tile_number += 1

            # ------------------------------------------------
            # Core region
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
            # ------------------------------------------------

            cy0 = max(
                0,
                y0 - context
            )

            cx0 = max(
                0,
                x0 - context
            )

            cy1 = min(
                height,
                y1 + context
            )

            cx1 = min(
                width,
                x1 + context
            )

            tile = features[
                :,
                cy0:cy1,
                cx0:cx1
            ]

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred = predict_tile(
                model,
                tile
            )

            # ------------------------------------------------
            # Extract core
            # ------------------------------------------------

            py0 = y0 - cy0
            px0 = x0 - cx0

            py1 = py0 + (
                y1 - y0
            )

            px1 = px0 + (
                x1 - x0
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

            print(
                f"\rTile "
                f"{tile_number}/{total_tiles}",
                end=""
            )

    print()

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    full_prediction = (
        prediction_sum
        /
        np.maximum(
            prediction_count,
            1.0
        )
    )

    return full_prediction


# ============================================================
# SAVE GEOTIFF
# ============================================================

def save_geotiff(
    output_path,
    prediction,
    profile
):

    print()
    print("=" * 70)
    print("SAVING PROBABILITY MAP")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(
            output_path
        ) or ".",
        exist_ok=True
    )

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate",
        nodata=0.0
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
            "Raw U-Net fault probability"
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
            "GeoDAWN standalone geothermal "
            "fault inference"
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
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output probability GeoTIFF"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Trained V1 U-Net checkpoint"
    )

    parser.add_argument(
        "--means",
        default=DEFAULT_MEANS,
        help="Feature means .npy"
    )

    parser.add_argument(
        "--stds",
        default=DEFAULT_STDS,
        help="Feature standard deviations .npy"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GEODAWN INFERENCE")
    print("=" * 70)

    print(
        "Input :",
        args.input
    )

    print(
        "Model :",
        args.model
    )

    print(
        "Output:",
        args.output
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        args.model
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        features,
        profile,
        transform,
        crs,
        raster_mask
    ) = load_features(
        args.input,
        args.means,
        args.stds
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    prediction = run_inference(
        model,
        features
    )

    # --------------------------------------------------------
    # Mask invalid raster areas
    # --------------------------------------------------------

    prediction[
        ~raster_mask
    ] = 0.0

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_prediction = prediction[
        raster_mask
    ]

    print()
    print("=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    print(
        f"Min    : {valid_prediction.min():.6f}"
    )

    print(
        f"Median : {np.median(valid_prediction):.6f}"
    )

    print(
        f"Mean   : {valid_prediction.mean():.6f}"
    )

    print(
        f"P90    : "
        f"{np.percentile(valid_prediction, 90):.6f}"
    )

    print(
        f"P99    : "
        f"{np.percentile(valid_prediction, 99):.6f}"
    )

    print(
        f"Max    : {valid_prediction.max():.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_geotiff(
        args.output,
        prediction,
        profile
    )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Output:",
        args.output
    )


if __name__ == "__main__":
    main()