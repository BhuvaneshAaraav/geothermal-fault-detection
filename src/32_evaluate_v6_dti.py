import numpy as np
from pathlib import Path

# Import the VERIFIED competition metric
from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/v6_validation_probability.npy"


# ============================================================
# LOAD RASTER
# ============================================================

import rasterio

print("=" * 70)
print("V6 COMPETITION METRIC EVALUATION")
print("=" * 70)

print("\nLoading labels...")

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

print("Labels:", labels.shape)


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

val_coords = np.load(VAL_COORDS_PATH)

print("Validation coordinates:", val_coords.shape)


# ============================================================
# LOAD V6 PREDICTIONS
# ============================================================

pred = np.load(PRED_PATH)

print("Prediction array shape:", pred.shape)


# ============================================================
# RECONSTRUCT FULL RASTER
# ============================================================

H, W = labels.shape

full_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

full_ground_truth = np.zeros(
    (H, W),
    dtype=np.float64
)


# ------------------------------------------------------------
# Predictions correspond to val_coords
# ------------------------------------------------------------

if pred.ndim == 1:

    if len(pred) != len(val_coords):
        raise ValueError(
            f"Prediction count {len(pred)} does not match "
            f"validation coordinates {len(val_coords)}"
        )

    for i, (y, x) in enumerate(val_coords):
        full_prediction[y, x] = pred[i]

else:

    # If already full raster
    if pred.shape == labels.shape:
        full_prediction = pred.astype(np.float64)

    else:
        raise ValueError(
            f"Unexpected prediction shape: {pred.shape}"
        )


# ============================================================
# GROUND TRUTH
# ============================================================

for y, x in val_coords:
    full_ground_truth[y, x] = (
        labels[y, x] > 0
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK")
print("=" * 70)

print(
    "Evaluation pixels:",
    np.sum(full_ground_truth * 0 + (full_prediction > 0))
)

print(
    "Ground-truth fault pixels:",
    int(np.sum(full_ground_truth))
)

print(
    "Prediction min:",
    full_prediction.min()
)

print(
    "Prediction max:",
    full_prediction.max()
)

print(
    "Prediction mean:",
    full_prediction[val_coords[:, 0], val_coords[:, 1]].mean()
)


# ============================================================
# COMPETITION METRIC
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING VERIFIED COMPETITION DTI")
print("=" * 70)

score = fast_metric(
    full_ground_truth,
    full_prediction
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("V6 COMPETITION RESULT")
print("=" * 70)

print(f"\nDistance-Weighted Tversky: {score:.9f}")

print("\nKnown benchmarks:")
print("----------------------------------------")
print("V1 raw DTI       : 0.146870")
print("V1 calibrated    : 0.180297*")
print("V2 raw DTI       : 0.109588")
print("V3 raw DTI       : 0.025407")
print("V4 approximate   : 0.079696†")
print(f"V6 exact DTI     : {score:.9f}")

print("\n* Validation-tuned calibration")
print("† V4 used a different implementation,")
print("  so it is not an exact comparison.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)