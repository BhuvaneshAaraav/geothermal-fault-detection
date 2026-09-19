import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/v7_validation_probability.npy"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V7 EXACT COMPETITION DTI")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

val_coords = np.load(
    VAL_COORDS_PATH
)

prediction = np.load(
    PRED_PATH
).astype(np.float64)

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]


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
# CANONICAL PREDICTION
# ============================================================

test_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

test_prediction[ys, xs] = (
    prediction[ys, xs]
)


# ============================================================
# EXACT METRIC
# ============================================================

print("\nEvaluation pixels:", len(val_coords))
print("Fault pixels:", int(gt.sum()))

print("\nCalculating exact competition DTI...")

score = fast_metric(
    gt,
    test_prediction
)


# ============================================================
# BASIC DIAGNOSTICS
# ============================================================

values = prediction[
    ys,
    xs
]

tp_pixels = np.sum(
    (
        values > 0.5
    )
    &
    (
        labels[ys, xs] > 0
    )
)

predicted_pixels = np.sum(
    values >= 0.5
)

precision = (
    tp_pixels / predicted_pixels
    if predicted_pixels > 0
    else 0.0
)

recall = (
    tp_pixels / gt.sum()
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("V7 EXACT DTI RESULT")
print("=" * 70)

print(
    f"\nV7 raw DTI       : {score:.9f}"
)

print(
    f"V1 raw DTI       : 0.148330184"
)

print(
    f"V1 calibrated DTI: 0.206570046"
)

print(
    f"\nDifference vs V1 raw:"
    f" {score - 0.148330184:+.9f}"
)

print(
    f"Difference vs V1 best:"
    f" {score - 0.206570046:+.9f}"
)

print(
    f"\nPixels >= 0.5    : {predicted_pixels:,}"
)

print(
    f"TP at >= 0.5     : {int(tp_pixels):,}"
)

print(
    f"Precision         : {precision:.5f}"
)

print(
    f"Recall            : {recall:.5f}"
)

print("\nDONE")