import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_DIR = "outputs/v1_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same validation blocks used everywhere
VALIDATION_BLOCKS = [1, 7, 9, 10]

THRESHOLD = 0.4


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

pred = np.load(PRED_PATH).astype(np.float32)

with rasterio.open(LABELS_PATH) as src:
    labels = src.read(1)

with rasterio.open(FEATURES_PATH) as src:
    valid_mask = src.read_masks(1) > 0

height, width = labels.shape

print("Prediction:", pred.shape)
print("Labels    :", labels.shape)
print("Valid     :", valid_mask.sum())


# ============================================================
# BUILD VALIDATION MASK
# ============================================================

print()
print("=" * 70)
print("BUILDING VALIDATION MASK")
print("=" * 70)

n_blocks_y = 4
n_blocks_x = 4

block_h = height // n_blocks_y
block_w = width // n_blocks_x

validation_mask = np.zeros(
    (height, width),
    dtype=bool
)

for block_id in VALIDATION_BLOCKS:

    by = block_id // n_blocks_x
    bx = block_id % n_blocks_x

    y0 = by * block_h
    y1 = (
        (by + 1) * block_h
        if by < n_blocks_y - 1
        else height
    )

    x0 = bx * block_w
    x1 = (
        (bx + 1) * block_w
        if bx < n_blocks_x - 1
        else width
    )

    validation_mask[y0:y1, x0:x1] = True


evaluation_mask = validation_mask & valid_mask

gt = (labels > 0) & evaluation_mask


# ============================================================
# THRESHOLD MASKS
# ============================================================

prediction_positive = (
    pred >= THRESHOLD
) & evaluation_mask

true_positive = (
    prediction_positive &
    gt
)

false_positive = (
    prediction_positive &
    ~gt
)

false_negative = (
    ~prediction_positive &
    gt
)


print()
print("Validation pixels :", validation_mask.sum())
print("Evaluation pixels :", evaluation_mask.sum())
print("Ground-truth faults:", gt.sum())
print("Predicted positive :", prediction_positive.sum())
print("True positives     :", true_positive.sum())
print("False positives    :", false_positive.sum())
print("False negatives    :", false_negative.sum())


# ============================================================
# HELPER
# ============================================================

def save_image(
    data,
    filename,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    interpolation="nearest"
):

    plt.figure(figsize=(14, 9))

    plt.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation
    )

    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.colorbar()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


# ============================================================
# 1. GROUND TRUTH
# ============================================================

gt_display = np.zeros(
    (height, width),
    dtype=np.float32
)

gt_display[gt] = 1.0

save_image(
    gt_display,
    "01_ground_truth.png",
    "Validation Ground-Truth Faults",
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# 2. RAW PROBABILITY MAP
# ============================================================

prob_display = pred.copy()

prob_display[~evaluation_mask] = np.nan

save_image(
    prob_display,
    "02_probability_map.png",
    "V1 Probability Map",
    cmap="viridis",
    vmin=0,
    vmax=1
)


# ============================================================
# 3. THRESHOLD MAP
# ============================================================

threshold_display = np.zeros(
    (height, width),
    dtype=np.float32
)

threshold_display[
    prediction_positive
] = 1.0

threshold_display[
    ~evaluation_mask
] = np.nan

save_image(
    threshold_display,
    "03_threshold_040.png",
    f"V1 Predictions >= {THRESHOLD}",
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# 4. TRUE POSITIVES
# ============================================================

tp_display = np.zeros(
    (height, width),
    dtype=np.float32
)

tp_display[true_positive] = 1.0
tp_display[~evaluation_mask] = np.nan

save_image(
    tp_display,
    "04_true_positives.png",
    f"True Positives >= {THRESHOLD}",
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# 5. FALSE POSITIVES
# ============================================================

fp_display = np.zeros(
    (height, width),
    dtype=np.float32
)

fp_display[false_positive] = 1.0
fp_display[~evaluation_mask] = np.nan

save_image(
    fp_display,
    "05_false_positives.png",
    f"False Positives >= {THRESHOLD}",
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# 6. FALSE NEGATIVES
# ============================================================

fn_display = np.zeros(
    (height, width),
    dtype=np.float32
)

fn_display[false_negative] = 1.0
fn_display[~evaluation_mask] = np.nan

save_image(
    fn_display,
    "06_false_negatives.png",
    f"False Negatives < {THRESHOLD}",
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# 7. ERROR MAP
#
# 0 = background
# 1 = TP
# 2 = FP
# 3 = FN
# ============================================================

error_map = np.zeros(
    (height, width),
    dtype=np.float32
)

error_map[true_positive] = 1
error_map[false_positive] = 2
error_map[false_negative] = 3

error_map[~evaluation_mask] = np.nan

save_image(
    error_map,
    "07_error_map.png",
    "V1 Error Map: TP / FP / FN",
    cmap="viridis",
    vmin=0,
    vmax=3
)


# ============================================================
# 8. HIGH CONFIDENCE PREDICTIONS
# ============================================================

high_confidence = (
    (pred >= 0.8) &
    evaluation_mask
)

high_conf_display = np.zeros(
    (height, width),
    dtype=np.float32
)

high_conf_display[
    high_confidence
] = pred[high_confidence]

high_conf_display[
    ~evaluation_mask
] = np.nan

save_image(
    high_conf_display,
    "08_high_confidence.png",
    "V1 High-Confidence Predictions (>= 0.8)",
    cmap="hot",
    vmin=0.8,
    vmax=1.0
)


# ============================================================
# 9. HIGH-CONFIDENCE FALSE POSITIVES
# ============================================================

high_fp = (
    (pred >= 0.8) &
    evaluation_mask &
    ~gt
)

high_fp_display = np.zeros(
    (height, width),
    dtype=np.float32
)

high_fp_display[
    high_fp
] = pred[high_fp]

high_fp_display[
    ~evaluation_mask
] = np.nan

save_image(
    high_fp_display,
    "09_high_confidence_false_positives.png",
    "V1 High-Confidence False Positives (>= 0.8)",
    cmap="hot",
    vmin=0.8,
    vmax=1.0
)


# ============================================================
# 10. FALSE POSITIVE DENSITY
#
# Useful for seeing large spatial clusters.
# ============================================================

from scipy.ndimage import uniform_filter

fp_float = false_positive.astype(np.float32)

fp_density = uniform_filter(
    fp_float,
    size=31,
    mode="constant"
)

fp_density[~evaluation_mask] = np.nan

save_image(
    fp_density,
    "10_false_positive_density.png",
    "False-Positive Spatial Density (31x31)",
    cmap="hot",
    vmin=0,
    vmax=np.nanpercentile(fp_density, 99)
)


# ============================================================
# 11. FALSE NEGATIVE DENSITY
# ============================================================

fn_float = false_negative.astype(np.float32)

fn_density = uniform_filter(
    fn_float,
    size=31,
    mode="constant"
)

fn_density[~evaluation_mask] = np.nan

save_image(
    fn_density,
    "11_false_negative_density.png",
    "False-Negative Spatial Density (31x31)",
    cmap="hot",
    vmin=0,
    vmax=np.nanpercentile(fn_density, 99)
)


# ============================================================
# 12. TOP HIGH-CONFIDENCE FP CLUSTERS
# ============================================================

print()
print("=" * 70)
print("HIGH-CONFIDENCE FALSE-POSITIVE LOCATIONS")
print("=" * 70)

ys, xs = np.where(high_fp)

if len(xs) == 0:

    print("No high-confidence false positives.")

else:

    scores = pred[ys, xs]

    order = np.argsort(scores)[::-1]

    print(
        f"Found {len(xs):,} false-positive pixels with "
        f"probability >= 0.8"
    )

    print()
    print("Top 50:")

    for rank, idx in enumerate(order[:50], 1):

        print(
            f"{rank:2d}. "
            f"row={ys[idx]:4d} "
            f"col={xs[idx]:4d} "
            f"prob={scores[idx]:.6f}"
        )


# ============================================================
# 13. SAVE ERROR COORDINATES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "false_positive_coords.npy"
    ),
    np.column_stack(
        np.where(false_positive)
    )
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "false_negative_coords.npy"
    ),
    np.column_stack(
        np.where(false_negative)
    )
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "high_confidence_fp_coords.npy"
    ),
    np.column_stack(
        np.where(high_fp)
    )
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    "All images saved to:",
    OUTPUT_DIR
)