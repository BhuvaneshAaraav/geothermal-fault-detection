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

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

K = 91_000
POWER = 0.001
RADIUS = 1


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("EXACT SCRIPT 51 POST-PROCESSING VALIDATION")
print("=" * 70)

with rasterio.open(PREDICTION_RASTER) as src:
    prediction = src.read(1).astype(np.float64)

with rasterio.open(LABEL_RASTER) as src:
    labels = src.read(1)

H, W = labels.shape

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print()
print("Prediction shape :", prediction.shape)
print("Label shape      :", labels.shape)
print("Validation pixels:", f"{len(val_coords):,}")


# ============================================================
# CANONICAL GT
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

GT_POSITIVES = int(gt.sum())

print(
    "Validation GT positives:",
    f"{GT_POSITIVES:,}"
)


# ============================================================
# EXACT SCRIPT 51 SPATIAL THINNING
# ============================================================

def spatial_thin(
    scores,
    coords,
    K,
    radius
):

    # EXACT:
    # argsort ALL validation scores
    order = np.argsort(scores)[::-1]

    selected = []

    occupied = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    for idx in order:

        y, x = coords[idx]

        y0 = max(
            0,
            y - radius
        )

        y1 = min(
            H,
            y + radius + 1
        )

        x0 = max(
            0,
            x - radius
        )

        x1 = min(
            W,
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
        ] = 1

        # IMPORTANT:
        # K is checked AFTER spatial acceptance
        if len(selected) >= K:
            break

    return np.array(
        selected,
        dtype=np.int64
    )


# ============================================================
# VALIDATION SCORES
# ============================================================

val_prediction = prediction[
    ys,
    xs
]

print()
print("=" * 70)
print("RAW PREDICTION")
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

raw_map[
    ys,
    xs
] = val_prediction

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
    "Expected:",
    "0.148330184"
)


# ============================================================
# POWER TRANSFORMATION
# ============================================================

scores = np.power(
    np.clip(
        val_prediction,
        0.0,
        1.0
    ),
    POWER
)


# ============================================================
# EXACT SCRIPT 51 SELECTION
#
# NO TOP-K BEFORE THINNING
# ============================================================

selected_idx = spatial_thin(
    scores,
    val_coords,
    K,
    RADIUS
)


# ============================================================
# SELECTED COORDINATES
# ============================================================

selected_ys = ys[
    selected_idx
]

selected_xs = xs[
    selected_idx
]

selected_scores = scores[
    selected_idx
]


# ============================================================
# FINAL PREDICTION MAP
#
# IMPORTANT:
# Script 51 puts the POWER-TRANSFORMED SCORE into the
# competition prediction map.
# ============================================================

test_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

test_prediction[
    selected_ys,
    selected_xs
] = selected_scores


# ============================================================
# EXACT METRIC
# ============================================================

dti = fast_metric(
    gt,
    test_prediction
)


# ============================================================
# STATISTICS
# ============================================================

selected_count = len(
    selected_idx
)

tp = int(
    np.sum(
        gt[
            selected_ys,
            selected_xs
        ] > 0
    )
)

fp = (
    selected_count
    -
    tp
)

precision = (
    tp / selected_count
    if selected_count > 0
    else 0.0
)

recall = (
    tp / GT_POSITIVES
    if GT_POSITIVES > 0
    else 0.0
)

threshold = (
    val_prediction[
        selected_idx[-1]
    ]
    if selected_count > 0
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("EXACT SCRIPT 51 RESULTS")
print("=" * 70)

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
print("Candidates")
print("--------------------------------------")

print(
    "Selected:",
    f"{selected_count:,}"
)

print(
    "Threshold:",
    f"{threshold:.9f}"
)

print()
print("Classification")
print("--------------------------------------")

print(
    "GT positives:",
    f"{GT_POSITIVES:,}"
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
print("=" * 70)
print("FINAL EXACT DTI")
print("=" * 70)

print(
    f"{dti:.9f}"
)

print()
print("Verified Script 51 champion:")
print("0.239444373")

print()
print("=" * 70)
print("DONE")
print("=" * 70)