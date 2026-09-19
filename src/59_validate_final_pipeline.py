import os
import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

PREDICTION_RASTER = (
    "outputs/final_test/raw_fault_probability.tif"
)

LABEL_RASTER = (
    "data/raw/Training_fault_labels.tif"
)

VAL_COORDS = (
    "data/processed/unet/val_coords.npy"
)

K = 91000
POWER = 0.001
RADIUS = 1


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

    order = np.argsort(-scores)

    occupied = np.zeros(
        (height, width),
        dtype=bool
    )

    selected = []

    for idx in order:

        y = int(rows[idx])
        x = int(cols[idx])

        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)

        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)

        if occupied[y0:y1, x0:x1].any():
            continue

        selected.append(idx)

        occupied[y0:y1, x0:x1] = True

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("GEODAWN FINAL PIPELINE VALIDATION")
print("=" * 70)


# ============================================================
# LOAD PREDICTION
# ============================================================

with rasterio.open(PREDICTION_RASTER) as src:

    prediction = src.read(1).astype(
        np.float64
    )

    raster_shape = prediction.shape

print()
print("Prediction raster:")
print("Shape:", raster_shape)


# ============================================================
# LOAD LABELS
# ============================================================

with rasterio.open(LABEL_RASTER) as src:

    labels = src.read(1)

print()
print("Label raster:")
print("Shape:", labels.shape)

if labels.shape != prediction.shape:

    raise ValueError(
        "Prediction and label raster shapes differ."
    )


H, W = labels.shape


# ============================================================
# LOAD CANONICAL VALIDATION COORDINATES
# ============================================================

val_coords = np.load(
    VAL_COORDS
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print()
print("Canonical validation pixels:")
print(f"{len(val_coords):,}")


# ============================================================
# CANONICAL GROUND TRUTH
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_prediction = prediction[
    ys,
    xs
]

print()
print("=" * 70)
print("VALIDATION PREDICTION")
print("=" * 70)

print(
    "Min    :",
    f"{val_prediction.min():.9f}"
)

print(
    "Median :",
    f"{np.median(val_prediction):.9f}"
)

print(
    "Mean   :",
    f"{val_prediction.mean():.9f}"
)

print(
    "P90    :",
    f"{np.percentile(val_prediction, 90):.9f}"
)

print(
    "P99    :",
    f"{np.percentile(val_prediction, 99):.9f}"
)

print(
    "Max    :",
    f"{val_prediction.max():.9f}"
)


# ============================================================
# RAW DTI
# ============================================================

raw_map = np.zeros(
    (H, W),
    dtype=np.float64
)

raw_map[ys, xs] = val_prediction

raw_dti = fast_metric(
    gt,
    raw_map
)

print()
print("=" * 70)
print("RAW EXACT DTI")
print("=" * 70)

print(
    f"{raw_dti:.9f}"
)

print(
    "Expected V1 raw:",
    "0.148330184"
)


# ============================================================
# TOP-K
# ============================================================

print()
print("=" * 70)
print("TOP-K + POWER + SPATIAL THINNING")
print("=" * 70)

# Only validation pixels
scores = val_prediction.copy()

# Top-K
actual_K = min(
    K,
    len(scores)
)

top_idx = np.argpartition(
    scores,
    -actual_K
)[-actual_K:]

candidate_scores = scores[
    top_idx
]

candidate_rows = ys[
    top_idx
]

candidate_cols = xs[
    top_idx
]


# ============================================================
# POWER TRANSFORMATION
# ============================================================

ranking_scores = np.power(
    candidate_scores,
    POWER
)


# ============================================================
# SORT
# ============================================================

order = np.argsort(
    -ranking_scores
)

candidate_rows = candidate_rows[
    order
]

candidate_cols = candidate_cols[
    order
]

candidate_scores = candidate_scores[
    order
]

ranking_scores = ranking_scores[
    order
]


# ============================================================
# SPATIAL THINNING
# ============================================================

selected = spatial_thin(
    candidate_rows,
    candidate_cols,
    ranking_scores,
    H,
    W,
    RADIUS
)

selected_rows = candidate_rows[
    selected
]

selected_cols = candidate_cols[
    selected
]

selected_scores = candidate_scores[
    selected
]

selected_ranking = ranking_scores[
    selected
]


# ============================================================
# CREATE FINAL MAP
# ============================================================

final_map = np.zeros(
    (H, W),
    dtype=np.float64
)

final_map[
    selected_rows,
    selected_cols
] = selected_scores


# ============================================================
# EXACT DTI
# ============================================================

final_dti = fast_metric(
    gt,
    final_map
)


# ============================================================
# STATISTICS
# ============================================================

positive_gt = int(
    gt.sum()
)

tp = int(
    np.sum(
        (final_map > 0)
        &
        (gt > 0)
    )
)

selected_count = len(
    selected_rows
)

fp = selected_count - tp

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
print("Configuration")
print("--------------------------------------")

print(
    "K:",
    f"{K:,}"
)

print(
    "Power:",
    POWER
)

print(
    "Radius:",
    RADIUS,
    "pixel"
)

print()
print("Results")
print("--------------------------------------")

print(
    "Initial candidates:",
    f"{actual_K:,}"
)

print(
    "Final candidates:",
    f"{selected_count:,}"
)

print(
    "Ground-truth faults:",
    f"{positive_gt:,}"
)

print(
    "TP:",
    f"{tp:,}"
)

print(
    "FP:",
    f"{fp:,}"
)

print(
    "Precision:",
    f"{precision:.6f}"
)

print(
    "Recall:",
    f"{recall:.6f}"
)

print()
print(
    "FINAL EXACT DTI:",
    f"{final_dti:.9f}"
)

print(
    "Previous champion:",
    "0.239444373"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)