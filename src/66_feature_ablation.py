"""
GeoDAWN — V1 U-Net Feature Ablation

For each of the 19 input bands:

    1. Load the exact normalized feature raster.
    2. Replace ONE band with zero.
       Zero corresponds approximately to the training mean
       after normalization.
    3. Run the exact V1 U-Net inference procedure.
    4. Evaluate the canonical held-out validation DTI.
    5. Compare against the original model.

IMPORTANT
---------
This is model ablation, not causal inference.

A large DTI decrease after removing a band indicates that
the trained model's performance depends on information from
that band under this ablation experiment.

It does NOT prove that the geological phenomenon itself is
caused by that feature.

Expected runtime:
-----------------
This runs inference approximately 20 times:
    1 baseline + 19 ablations

On an M1 MacBook this can take considerable time.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = (
    "data/raw/training_features.tif"
)

LABEL_PATH = (
    "data/raw/Training_fault_labels.tif"
)

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

MEANS_PATH = (
    "data/processed/cnn/feature_means.npy"
)

STDS_PATH = (
    "data/processed/cnn/feature_stds.npy"
)

CHECKPOINT_PATH = (
    "models/fault_tversky_unet_best.pt"
)

OUTPUT_DIR = (
    "outputs/final_test/feature_ablation"
)


# ============================================================
# INFERENCE CONFIG
# ============================================================

PATCH_SIZE = 31

CONTEXT = PATCH_SIZE // 2

TILE_SIZE = 256

OVERLAP = 32

STEP = TILE_SIZE - OVERLAP

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VERIFIED METRIC
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:

    sys.path.insert(
        0,
        SRC_DIR
    )

from src19_metric import fast_metric


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("GEODAWN V1 U-NET FEATURE ABLATION")
print("=" * 75)

print()
print(
    "Device:",
    DEVICE
)


# ============================================================
# U-NET ARCHITECTURE
# ============================================================

class ConvBlock(
    nn.Module
):

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

    def forward(
        self,
        x
    ):

        return self.block(x)


class FaultUNet(
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

        self.pool1 = nn.MaxPool2d(
            2
        )

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(
            2
        )

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(
            2
        )

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

    def forward(
        self,
        x
    ):

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
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(d3)

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

        d1 = self.up1(d2)

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
# LOAD MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING V1 CHECKPOINT")
print("=" * 75)

model = FaultUNet(
    in_channels=19
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.to(
    DEVICE
)

model.eval()

print(
    "Checkpoint loaded:"
)

print(
    CHECKPOINT_PATH
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 75)
print("LOADING FEATURES")
print("=" * 75)

with rasterio.open(
    FEATURE_PATH
) as src:

    data = src.read().astype(
        np.float32
    )

    feature_mask = (
        src.read_masks(1) > 0
    )

    descriptions = list(
        src.descriptions
    )

    raster_profile = src.profile

n_bands, height, width = (
    data.shape
)

print(
    "Shape:",
    data.shape
)

print(
    "Valid pixels:",
    f"{feature_mask.sum():,}"
)


# ============================================================
# BAND NAMES
# ============================================================

band_names = []

for i in range(
    n_bands
):

    name = descriptions[i]

    if name is None or name == "":

        name = f"band_{i+1}"

    band_names.append(
        name
    )


# ============================================================
# NORMALIZATION
# ============================================================

print()
print("=" * 75)
print("NORMALIZING FEATURES")
print("=" * 75)

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
    "Means:",
    means.shape
)

print(
    "Stds:",
    stds.shape
)


for band in range(
    n_bands
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

    data[
        band,
        invalid
    ] = means[band]

    data[
        band
    ] -= means[band]

    data[
        band
    ] /= max(
        float(stds[band]),
        1e-8
    )


np.clip(
    data,
    -10,
    10,
    out=data
)

data[
    :,
    ~feature_mask
] = 0


print(
    "Normalized range:",
    float(data.min()),
    "to",
    float(data.max())
)


# ============================================================
# VALIDATION COORDINATES
# ============================================================

print()
print("=" * 75)
print("LOADING VALIDATION SET")
print("=" * 75)

val_coords = np.load(
    VAL_COORDS_PATH
).astype(
    np.int64
)

ys = val_coords[:, 0]

xs = val_coords[:, 1]

print(
    "Validation pixels:",
    f"{len(val_coords):,}"
)


# ============================================================
# LABELS
# ============================================================

with rasterio.open(
    LABEL_PATH
) as src:

    labels = src.read(1)


gt = np.zeros(
    (height, width),
    dtype=np.float64
)

gt[
    ys,
    xs
] = (
    labels[
        ys,
        xs
    ] > 0
).astype(
    np.float64
)

print(
    "Validation fault pixels:",
    f"{int(gt.sum()):,}"
)


# ============================================================
# INFERENCE FUNCTION
# ============================================================

@torch.inference_mode()
def predict_raster(
    feature_data
):

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    positions_y = range(
        0,
        height,
        STEP
    )

    positions_x = range(
        0,
        width,
        STEP
    )

    total_tiles = (
        len(
            range(
                0,
                height,
                STEP
            )
        )
        *
        len(
            range(
                0,
                width,
                STEP
            )
        )
    )

    tile_number = 0

    for y0 in positions_y:

        y1 = min(
            y0 + TILE_SIZE,
            height
        )

        for x0 in positions_x:

            x1 = min(
                x0 + TILE_SIZE,
                width
            )

            tile_number += 1

            # EXACT Script 15 context behavior.
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

            tile = feature_data[
                :,
                cy0:cy1,
                cx0:cx1
            ]

            tensor = torch.from_numpy(
                tile
            ).unsqueeze(
                0
            ).to(
                DEVICE,
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

            py0 = y0 - cy0
            px0 = x0 - cx0

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

            prediction_sum[
                y0:y1,
                x0:x1
            ] += core_prediction

            prediction_count[
                y0:y1,
                x0:x1
            ] += 1.0

    full_prediction = (
        prediction_sum
        /
        np.maximum(
            prediction_count,
            1.0
        )
    )

    full_prediction[
        ~feature_mask
    ] = 0.0

    full_prediction = np.clip(
        full_prediction,
        0,
        1
    ).astype(
        np.float32
    )

    return full_prediction


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 75)
print("BASELINE INFERENCE")
print("=" * 75)

start = time.time()

baseline_prediction = predict_raster(
    data
)

baseline_time = (
    time.time()
    -
    start
)

baseline_dti = fast_metric(
    gt,
    baseline_prediction.astype(
        np.float64
    )
)

print()
print(
    "Baseline DTI:",
    f"{baseline_dti:.9f}"
)

print(
    "Inference time:",
    f"{baseline_time / 60:.2f} minutes"
)


# ============================================================
# SAVE BASELINE
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "baseline_prediction.npy"
    ),
    baseline_prediction
)


# ============================================================
# ABLATION LOOP
# ============================================================

results = []

print()
print("=" * 75)
print("STARTING 19-BAND ABLATION")
print("=" * 75)

for band in range(
    n_bands
):

    print()
    print("-" * 75)

    print(
        f"ABLATION {band+1}/{n_bands}"
    )

    print(
        "Band:",
        band + 1
    )

    print(
        "Feature:",
        band_names[band]
    )

    # --------------------------------------------------------
    # COPY NORMALIZED DATA
    # --------------------------------------------------------

    ablated_data = data.copy()

    # Zero = mean feature after normalization.
    ablated_data[
        band
    ] = 0.0

    print(
        "Band replaced with normalized mean (0)."
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start = time.time()

    ablated_prediction = predict_raster(
        ablated_data
    )

    elapsed = (
        time.time()
        -
        start
    )

    # --------------------------------------------------------
    # METRIC
    # --------------------------------------------------------

    ablated_dti = fast_metric(
        gt,
        ablated_prediction.astype(
            np.float64
        )
    )

    delta = (
        ablated_dti
        -
        baseline_dti
    )

    degradation = (
        baseline_dti
        -
        ablated_dti
    )

    relative_change = (
        degradation
        /
        baseline_dti
        *
        100.0
    )

    print()
    print(
        "Ablated DTI:",
        f"{ablated_dti:.9f}"
    )

    print(
        "DTI change:",
        f"{delta:+.9f}"
    )

    print(
        "DTI degradation:",
        f"{degradation:+.9f}"
    )

    print(
        "Relative degradation:",
        f"{relative_change:+.2f}%"
    )

    print(
        "Time:",
        f"{elapsed / 60:.2f} minutes"
    )

    results.append({

        "band":
            band + 1,

        "feature":
            band_names[band],

        "baseline_dti":
            baseline_dti,

        "ablated_dti":
            ablated_dti,

        "dti_change":
            delta,

        "dti_degradation":
            degradation,

        "relative_degradation_percent":
            relative_change,

        "inference_time_minutes":
            elapsed / 60.0
    })

    # --------------------------------------------------------
    # SAVE ABLATED PREDICTION
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"ablation_band_{band+1:02d}.npy"
        ),
        ablated_prediction
    )

    # Explicit cleanup
    del ablated_data
    del ablated_prediction

    if DEVICE.type == "mps":

        torch.mps.empty_cache()


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "dti_degradation",
    ascending=False
).reset_index(
    drop=True
)

results_df[
    "importance_rank"
] = np.arange(
    1,
    len(results_df) + 1
)


# ============================================================
# SAVE
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("FEATURE ABLATION RESULTS")
print("=" * 75)

print(
    results_df[
        [
            "importance_rank",
            "band",
            "feature",
            "ablated_dti",
            "dti_degradation",
            "relative_degradation_percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BAR PLOT
# ============================================================

plot_df = results_df.sort_values(
    "dti_degradation",
    ascending=True
)

plt.figure(
    figsize=(12, 9)
)

plt.barh(
    plot_df["feature"],
    plot_df["dti_degradation"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "DTI degradation after feature ablation"
)

plt.ylabel(
    "Geophysical feature"
)

plt.title(
    "GeoDAWN V1 — Feature Ablation Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_ablation_importance.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "ablation_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "GeoDAWN V1 Feature Ablation\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Baseline validation DTI: "
        f"{baseline_dti:.9f}\n\n"
    )

    f.write(
        "Feature importance ranking\n"
    )

    f.write(
        "---------------------------\n"
    )

    for _, row in results_df.iterrows():

        f.write(
            f"{int(row['importance_rank']):2d}. "
            f"{row['feature']} | "
            f"ablated DTI="
            f"{row['ablated_dti']:.9f} | "
            f"degradation="
            f"{row['dti_degradation']:.9f} | "
            f"relative="
            f"{row['relative_degradation_percent']:.2f}%\n"
        )

    f.write(
        "\nInterpretation\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "Feature importance here is defined by the "
        "change in validation DTI after replacing one "
        "normalized input band with zero while keeping "
        "the trained model unchanged.\n\n"
    )

    f.write(
        "A larger positive DTI degradation means that "
        "the model's validation performance was more "
        "affected by removing that band's information "
        "under this experiment.\n\n"
    )

    f.write(
        "This is model-specific ablation evidence and "
        "should not be interpreted as proof of geological "
        "causation or as an independent estimate of "
        "physical feature importance.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("FEATURE ABLATION COMPLETE")
print("=" * 75)

print()
print(
    "Baseline DTI:",
    f"{baseline_dti:.9f}"
)

print(
    "Results:",
    results_path
)

print(
    "Plot:",
    os.path.join(
        OUTPUT_DIR,
        "feature_ablation_importance.png"
    )
)

print()
print(
    "Output directory:",
    OUTPUT_DIR
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)