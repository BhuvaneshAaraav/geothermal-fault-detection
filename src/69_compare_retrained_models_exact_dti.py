import os
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

FEATURE_RASTER = "data/raw/training_features.tif"
LABEL_RASTER = "data/raw/Training_fault_labels.tif"

VAL_COORDS = "data/processed/unet/val_coords.npy"

MEANS_FILE = "data/processed/cnn/feature_means.npy"
STDS_FILE = "data/processed/cnn/feature_stds.npy"

MODEL_DIR = "outputs/final_test/reduced_feature_models"

OUTPUT_DIR = (
    "outputs/final_test/reduced_feature_models/exact_dti"
)

PATCH_SIZE = 31

K = 91000
POWER = 0.001
RADIUS = 1

BATCH_SIZE = 16

SEED = 42


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    "V1_19": list(range(19)),

    "V1_no3": [
        i for i in range(19)
        if i != 2
    ],

    "V1_no4": [
        i for i in range(19)
        if i != 3
    ],

    "V1_no7": [
        i for i in range(19)
        if i != 6
    ],
}


BAND_NAMES = [
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

    print("Using Apple MPS GPU")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("Using CUDA GPU")

else:

    DEVICE = torch.device("cpu")

    print("Using CPU")


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
                padding=1
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
                padding=1
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
        in_channels
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

        d3 = torch.nn.functional.interpolate(
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

        d2 = torch.nn.functional.interpolate(
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

        d1 = torch.nn.functional.interpolate(
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

class ValidationPatchDataset(Dataset):

    def __init__(
        self,
        features,
        coordinates,
        feature_indices
    ):

        self.features = features
        self.coordinates = coordinates
        self.feature_indices = feature_indices

        self.radius = PATCH_SIZE // 2

    def __len__(self):

        return len(self.coordinates)

    def __getitem__(self, idx):

        y, x = self.coordinates[idx]

        y = int(y)
        x = int(x)

        r = self.radius

        patch = self.features[
            self.feature_indices,
            y-r:y+r+1,
            x-r:x+r+1
        ]

        return torch.from_numpy(
            patch.copy()
        )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

val_coords = np.load(
    VAL_COORDS
)

print(
    "Validation coordinates:",
    val_coords.shape
)

print(
    "Validation pixels:",
    f"{len(val_coords):,}"
)


# ============================================================
# NORMALIZATION
# ============================================================

means = np.load(
    MEANS_FILE
).astype(np.float32)

stds = np.load(
    STDS_FILE
).astype(np.float32)


# ============================================================
# LOAD FEATURES
# ============================================================

print()
print("=" * 70)
print("LOADING FEATURES")
print("=" * 70)

with rasterio.open(
    FEATURE_RASTER
) as src:

    features = src.read(
        out_dtype="float32"
    )

    feature_mask = (
        src.read_masks(1) > 0
    )

print(
    "Feature shape:",
    features.shape
)


# ============================================================
# EXACT V1 NORMALIZATION
# ============================================================

for band in range(
    features.shape[0]
):

    band_data = features[band]

    invalid = (
        (~feature_mask)
        | (~np.isfinite(band_data))
    )

    band_data[invalid] = means[band]

    band_data -= means[band]

    band_data /= max(
        stds[band],
        1e-8
    )

    np.clip(
        band_data,
        -10.0,
        10.0,
        out=band_data
    )

features[
    :,
    ~feature_mask
] = 0.0

features = np.ascontiguousarray(
    features,
    dtype=np.float32
)

del feature_mask

print(
    "Normalization complete."
)


# ============================================================
# LOAD LABELS
# ============================================================

with rasterio.open(
    LABEL_RASTER
) as src:

    labels = src.read(
        1,
        out_dtype="uint8"
    )

labels = labels.astype(
    np.float64
)


# ============================================================
# CANONICAL GROUND TRUTH
# ============================================================

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

print()
print(
    "Validation ground-truth faults:",
    f"{int(gt.sum()):,}"
)


# ============================================================
# SPATIAL THINNING
# ============================================================

def spatial_thin(
    rows,
    cols,
    scores,
    height,
    width,
    radius
):

    order = np.argsort(
        -scores
    )

    occupied = np.zeros(
        (height, width),
        dtype=bool
    )

    selected = []

    for idx in order:

        y = int(rows[idx])
        x = int(cols[idx])

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

        selected.append(idx)

        occupied[
            y0:y1,
            x0:x1
        ] = True

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer_model(
    model,
    feature_indices
):

    dataset = ValidationPatchDataset(
        features,
        val_coords,
        feature_indices
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    predictions = np.zeros(
        len(val_coords),
        dtype=np.float64
    )

    model.eval()

    center = PATCH_SIZE // 2

    offset = 0

    for batch_idx, x in enumerate(
        loader
    ):

        x = x.to(
            DEVICE,
            dtype=torch.float32
        )

        logits = model(x)

        probabilities = torch.sigmoid(
            logits
        )

        center_prob = probabilities[
            :,
            0,
            center,
            center
        ]

        n = len(center_prob)

        predictions[
            offset:offset+n
        ] = (
            center_prob
            .detach()
            .cpu()
            .numpy()
        )

        offset += n

        if (
            batch_idx % 500 == 0
            or batch_idx == len(loader) - 1
        ):

            print(
                f"\rInference "
                f"{batch_idx + 1}/"
                f"{len(loader)}",
                end=""
            )

    print()

    return predictions


# ============================================================
# EVALUATE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = []


for experiment_name, feature_indices in EXPERIMENTS.items():

    print()
    print()
    print("=" * 70)
    print(
        f"MODEL: {experiment_name}"
    )
    print("=" * 70)

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"{experiment_name}_best.pt"
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "Input bands:",
        len(feature_indices)
    )

    removed = [
        BAND_NAMES[i]
        for i in range(19)
        if i not in feature_indices
    ]

    if removed:

        print(
            "Removed:",
            ", ".join(removed)
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = FaultSegmentationUNet(
        in_channels=len(feature_indices)
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Exact validation inference
    # --------------------------------------------------------

    val_prediction = infer_model(
        model,
        feature_indices
    )

    print()
    print(
        "Prediction statistics:"
    )

    print(
        f"Min    : "
        f"{val_prediction.min():.9f}"
    )

    print(
        f"Median : "
        f"{np.median(val_prediction):.9f}"
    )

    print(
        f"Mean   : "
        f"{val_prediction.mean():.9f}"
    )

    print(
        f"P90    : "
        f"{np.percentile(val_prediction, 90):.9f}"
    )

    print(
        f"P99    : "
        f"{np.percentile(val_prediction, 99):.9f}"
    )

    print(
        f"Max    : "
        f"{val_prediction.max():.9f}"
    )

    # --------------------------------------------------------
    # RAW DTI
    # --------------------------------------------------------

    raw_map = np.zeros(
        (H, W),
        dtype=np.float64
    )

    raw_map[
        ys,
        xs
    ] = val_prediction

    raw_dti = fast_metric(
        gt,
        raw_map
    )

    print()
    print(
        "RAW EXACT DTI:",
        f"{raw_dti:.9f}"
    )

    # --------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------

    actual_K = min(
        K,
        len(val_prediction)
    )

    top_idx = np.argpartition(
        val_prediction,
        -actual_K
    )[-actual_K:]

    candidate_scores = (
        val_prediction[top_idx]
    )

    candidate_rows = ys[
        top_idx
    ]

    candidate_cols = xs[
        top_idx
    ]

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    ranking_scores = np.power(
        candidate_scores,
        POWER
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    order = np.argsort(
        -ranking_scores
    )

    candidate_rows = (
        candidate_rows[order]
    )

    candidate_cols = (
        candidate_cols[order]
    )

    candidate_scores = (
        candidate_scores[order]
    )

    ranking_scores = (
        ranking_scores[order]
    )

    # --------------------------------------------------------
    # SPATIAL THINNING
    # --------------------------------------------------------

    selected = spatial_thin(
        candidate_rows,
        candidate_cols,
        ranking_scores,
        H,
        W,
        RADIUS
    )

    selected_rows = (
        candidate_rows[selected]
    )

    selected_cols = (
        candidate_cols[selected]
    )

    selected_scores = (
        candidate_scores[selected]
    )

    # --------------------------------------------------------
    # FINAL MAP
    # --------------------------------------------------------

    final_map = np.zeros(
        (H, W),
        dtype=np.float64
    )

    final_map[
        selected_rows,
        selected_cols
    ] = selected_scores

    # --------------------------------------------------------
    # EXACT DTI
    # --------------------------------------------------------

    final_dti = fast_metric(
        gt,
        final_map
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    positive_gt = int(
        gt.sum()
    )

    tp = int(
        np.sum(
            (final_map > 0)
            & (gt > 0)
        )
    )

    selected_count = len(
        selected_rows
    )

    fp = (
        selected_count
        - tp
    )

    precision = (
        tp / selected_count
        if selected_count
        else 0
    )

    recall = (
        tp / positive_gt
        if positive_gt
        else 0
    )

    print()
    print(
        "FINAL RESULTS"
    )
    print("-" * 70)

    print(
        f"Initial candidates : "
        f"{actual_K:,}"
    )

    print(
        f"Final candidates   : "
        f"{selected_count:,}"
    )

    print(
        f"Ground-truth faults: "
        f"{positive_gt:,}"
    )

    print(
        f"TP                 : "
        f"{tp:,}"
    )

    print(
        f"FP                 : "
        f"{fp:,}"
    )

    print(
        f"Precision          : "
        f"{precision:.6f}"
    )

    print(
        f"Recall             : "
        f"{recall:.6f}"
    )

    print()
    print(
        f"FINAL EXACT DTI    : "
        f"{final_dti:.9f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_validation_prediction.npy"
        ),
        val_prediction
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_final_map.npy"
        ),
        final_map
    )

    results.append({
        "model": experiment_name,
        "raw_dti": raw_dti,
        "final_dti": final_dti,
        "initial_candidates": actual_K,
        "final_candidates": selected_count,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall
    })


# ============================================================
# SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("EXACT DTI COMPARISON")
print("=" * 70)

for r in results:

    print(
        f"{r['model']:10s} "
        f"| Raw DTI = "
        f"{r['raw_dti']:.9f} "
        f"| Final DTI = "
        f"{r['final_dti']:.9f}"
    )

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

import csv

summary_path = os.path.join(
    OUTPUT_DIR,
    "exact_dti_comparison.csv"
)

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)


print()
print(
    "Saved:",
    summary_path
)

print()
print(
    "Previous canonical champion:",
    "0.239444373"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)