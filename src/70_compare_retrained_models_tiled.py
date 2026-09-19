import os

import numpy as np

import rasterio

import torch

import torch.nn as nn

# ============================================================

# CONFIG — EXACT SCRIPT 15 INFERENCE

# ============================================================

FEATURE_RASTER = "data/raw/training_features.tif"

LABEL_RASTER = "data/raw/Training_fault_labels.tif"

MEANS_FILE = "data/processed/cnn/feature_means.npy"

STDS_FILE = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "outputs/final_test/reduced_feature_models"

OUTPUT_DIR = "outputs/final_test/reduced_feature_models/tiled_inference"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE = 256

OVERLAP = 32

PATCH_SIZE = 31

RADIUS = 3

ALPHA = 0.2

BETA = 0.8

# ============================================================

# EXPERIMENTS

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

EXPERIMENTS = {

    "V1_19": list(range(19)),

    "V1_no3": [i for i in range(19) if i != 2],

    "V1_no4": [i for i in range(19) if i != 3],

    "V1_no7": [i for i in range(19) if i != 6],

}

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

# EXACT V1 U-NET

# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                3,

                padding=1,

                bias=False

            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(

                out_channels,

                out_channels,

                3,

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

            2,

            stride=2

        )

        self.dec3 = ConvBlock(

            256,

            128

        )

        self.up2 = nn.ConvTranspose2d(

            128,

            64,

            2,

            stride=2

        )

        self.dec2 = ConvBlock(

            128,

            64

        )

        self.up1 = nn.ConvTranspose2d(

            64,

            32,

            2,

            stride=2

        )

        self.dec1 = ConvBlock(

            64,

            32

        )

        self.final = nn.Conv2d(

            32,

            1,

            1

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

# LOAD RASTER

# ============================================================

print()

print("=" * 70)

print("LOADING RASTER")

print("=" * 70)

with rasterio.open(FEATURE_RASTER) as src:

    features = src.read(

        out_dtype="float32"

    )

    feature_mask = (

        src.read_masks(1) > 0

    )

height = features.shape[1]

width = features.shape[2]

print("Feature shape:", features.shape)

print("Raster:", width, "x", height)

# ============================================================

# NORMALIZATION — EXACT SCRIPT 15

# ============================================================

print()

print("NORMALIZING FEATURES")

means = np.load(

    MEANS_FILE

).astype(np.float32)

stds = np.load(

    STDS_FILE

).astype(np.float32)

for band in range(features.shape[0]):

    data = features[band]

    invalid = (

        (~feature_mask)

        | (~np.isfinite(data))

    )

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

features[:, ~feature_mask] = 0.0

features = np.ascontiguousarray(

    features,

    dtype=np.float32

)

# ============================================================

# LOAD LABELS

# ============================================================

print()

print("LOADING LABELS")

with rasterio.open(LABEL_RASTER) as src:

    labels = src.read(

        1,

        out_dtype="uint8"

    )

labels = labels.astype(

    np.float32

)

print(

    "Total fault pixels:",

    int(labels.sum())

)

# ============================================================

# CANONICAL VALIDATION MASK — EXACT SCRIPT 15

# ============================================================

rows_per_block = height // 4

cols_per_block = width // 4

validation_blocks = {

    1,

    7,

    9,

    10

}

validation_mask = np.zeros(

    (height, width),

    dtype=bool

)

for block_id in range(16):

    block_row = block_id // 4

    block_col = block_id % 4

    y0 = block_row * rows_per_block

    y1 = (

        height

        if block_row == 3

        else (block_row + 1) * rows_per_block

    )

    x0 = block_col * cols_per_block

    x1 = (

        width

        if block_col == 3

        else (block_col + 1) * cols_per_block

    )

    if block_id in validation_blocks:

        validation_mask[

            y0:y1,

            x0:x1

        ] = True

print()

print("Validation pixels:", int(validation_mask.sum()))

print(

    "Validation fault pixels:",

    int(labels[validation_mask].sum())

)

# ============================================================

# NUMPY SHIFT — EXACT SCRIPT 15

# ============================================================

def shift_zero_numpy(

    array,

    dy,

    dx

):

    h, w = array.shape

    result = np.zeros_like(

        array,

        dtype=np.float32

    )

    src_y0 = max(0, dy)

    src_y1 = min(h, h + dy)

    dst_y0 = max(0, -dy)

    dst_y1 = (

        dst_y0

        + src_y1

        - src_y0

    )

    src_x0 = max(0, dx)

    src_x1 = min(w, w + dx)

    dst_x0 = max(0, -dx)

    dst_x1 = (

        dst_x0

        + src_x1

        - src_x0

    )

    if (

        src_y1 > src_y0

        and src_x1 > src_x0

    ):

        result[

            dst_y0:dst_y1,

            dst_x0:dst_x1

        ] = array[

            src_y0:src_y1,

            src_x0:src_x1

        ]

    return result

# ============================================================

# EXACT FULL-RASTER DTI

# ============================================================

def calculate_dti(

    full_prediction

):

    p = np.where(

        validation_mask,

        full_prediction,

        0.0

    ).astype(np.float32)

    g = np.where(

        validation_mask,

        labels,

        0.0

    ).astype(np.float32)

    tp_match = np.zeros_like(

        p,

        dtype=np.float32

    )

    gt_influence = np.zeros_like(

        g,

        dtype=np.float32

    )

    for dy in range(

        -RADIUS,

        RADIUS + 1

    ):

        for dx in range(

            -RADIUS,

            RADIUS + 1

        ):

            distance = np.sqrt(

                dx * dx

                + dy * dy

            )

            if distance > RADIUS:

                continue

            kernel = (

                1.0

                - distance / RADIUS

            )

            shifted_prediction = (

                shift_zero_numpy(

                    p,

                    dy,

                    dx

                )

                * kernel

            )

            tp_match = np.maximum(

                tp_match,

                shifted_prediction

            )

            shifted_gt = (

                shift_zero_numpy(

                    g,

                    -dy,

                    -dx

                )

                * kernel

            )

            gt_influence = np.maximum(

                gt_influence,

                shifted_gt

            )

    tp = np.sum(

        g * tp_match

    )

    fn = np.sum(

        g * (1.0 - tp_match)

    )

    fp = np.sum(

        p * (1.0 - gt_influence)

    )

    denominator = (

        tp

        + ALPHA * fp

        + BETA * fn

    )

    score = (

        tp / denominator

        if denominator > 0

        else 0.0

    )

    return (

        float(score),

        float(tp),

        float(fp),

        float(fn)

    )

# ============================================================

# TILED INFERENCE

# ============================================================

def run_inference(

    model,

    feature_indices,

    name

):

    print()

    print("=" * 70)

    print("TILED INFERENCE:", name)

    print("=" * 70)

    print(

        "Input bands:",

        len(feature_indices)

    )

    print(

        "Bands:",

        ", ".join(

            FEATURE_NAMES[i]

            for i in feature_indices

        )

    )

    prediction_sum = np.zeros(

        (height, width),

        dtype=np.float32

    )

    prediction_count = np.zeros(

        (height, width),

        dtype=np.float32

    )

    context = PATCH_SIZE // 2

    step = TILE_SIZE - OVERLAP

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

        * len(tile_x_positions)

    )

    tile_number = 0

    @torch.no_grad()

    def predict_tile(tile):

        tensor = torch.from_numpy(

            tile

        ).unsqueeze(0).to(

            DEVICE,

            dtype=torch.float32

        )

        logits = model(tensor)

        probabilities = torch.sigmoid(

            logits

        )

        return probabilities[

            0,

            0

        ].cpu().numpy()

    for y0 in tile_y_positions:

        for x0 in tile_x_positions:

            tile_number += 1

            y1 = min(

                y0 + TILE_SIZE,

                height

            )

            x1 = min(

                x0 + TILE_SIZE,

                width

            )

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

                feature_indices,

                cy0:cy1,

                cx0:cx1

            ]

            pred = predict_tile(tile)

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

    full_prediction = (

        prediction_sum

        / np.maximum(

            prediction_count,

            1.0

        )

    )

    full_prediction[

        ~feature_mask

    ] = 0.0

    full_prediction = np.clip(

        full_prediction,

        0.0,

        1.0

    ).astype(np.float32)

    output_file = os.path.join(

        OUTPUT_DIR,

        f"{name}_full_prediction.npy"

    )

    np.save(

        output_file,

        full_prediction

    )

    score, tp, fp, fn = calculate_dti(

        full_prediction

    )

    val_pred = full_prediction[

        validation_mask

    ]

    print()

    print("-" * 70)

    print(name)

    print("-" * 70)

    print(

        "Checkpoint:",

        checkpoint_path

    )

    print(

        "Best training Tversky:",

        checkpoint.get(

            "val_tversky",

            "unknown"

        )

    )

    print(

        "Raw validation DTI:",

        f"{score:.9f}"

    )

    print(

        "TP weighted:",

        f"{tp:.6f}"

    )

    print(

        "FP weighted:",

        f"{fp:.6f}"

    )

    print(

        "FN weighted:",

        f"{fn:.6f}"

    )

    print(

        "Validation prediction mean:",

        f"{float(val_pred.mean()):.9f}"

    )

    print(

        "Validation prediction median:",

        f"{float(np.median(val_pred)):.9f}"

    )

    return score

# ============================================================

# RUN

# ============================================================

results = {}

for name, feature_indices in EXPERIMENTS.items():

    checkpoint_path = os.path.join(

        MODEL_DIR,

        f"{name}_best.pt"

    )

    print()

    print("=" * 70)

    print("LOADING:", checkpoint_path)

    print("=" * 70)

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE

    )

    model = FaultSegmentationUNet(

        in_channels=len(feature_indices)

    ).to(DEVICE)

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.eval()

    results[name] = run_inference(

        model,

        feature_indices,

        name

    )

    del model

    if DEVICE.type == "mps":

        torch.mps.empty_cache()

# ============================================================

# FINAL SUMMARY

# ============================================================

print()

print("=" * 70)

print("FINAL TILED-INFERENCE COMPARISON")

print("=" * 70)

for name, score in results.items():

    print(

        f"{name:10s} "

        f"raw_DTI = {score:.9f}"

    )

print()

print(

    "Current established production "

    "postprocessed DTI = 0.239444373"

)

print()

print(

    "Prediction files saved to:"

)

print(

    OUTPUT_DIR

)