import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# LOCKED V1_19 FINAL CONFIGURATION
# ============================================================

PRED_PATH = (
    "outputs/final_test/"
    "reduced_feature_models/"
    "tiled_inference/"
    "V1_19_full_prediction.npy"
)

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

K = 84_500

POWER = 0.00005

RADIUS = 1


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 75)
print("V1_19 FINAL REPRODUCIBILITY CHECK")
print("=" * 75)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    H, W = labels.shape


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]

xs = val_coords[:, 1]

print(
    f"Validation coordinates : {len(val_coords):,}"
)


# ============================================================
# GROUND TRUTH
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

GT_POSITIVES = int(
    gt.sum()
)

print(
    f"Validation GT positives : {GT_POSITIVES:,}"
)


# ============================================================
# LOAD V1_19 PREDICTION
# ============================================================

print()
print(
    "Loading:",
    PRED_PATH
)

prediction = np.load(
    PRED_PATH
)

print(
    "Prediction shape        :",
    prediction.shape
)


# ============================================================
# EXACT PREPROCESSING
# ============================================================

val_prediction = prediction[
    ys,
    xs
].astype(np.float64)

val_prediction = np.clip(
    val_prediction,
    0.0,
    1.0
)

scores = np.power(
    val_prediction,
    POWER
)


# ============================================================
# EXACT SPATIAL THINNING
# ============================================================

def spatial_thin(
    scores,
    coords,
    K,
    radius
):

    order = np.argsort(
        scores
    )[::-1]

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

        selected.append(
            idx
        )

        occupied[
            y0:y1,
            x0:x1
        ] = 1

        if len(selected) >= K:

            break

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# SELECT FINAL CANDIDATES
# ============================================================

selected = spatial_thin(
    scores,
    val_coords,
    K,
    RADIUS
)

sy = ys[selected]

sx = xs[selected]


# ============================================================
# FINAL PREDICTION MAP
# ============================================================

test_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

test_prediction[
    sy,
    sx
] = scores[
    selected
]


# ============================================================
# EXACT COMPETITION METRIC
# ============================================================

dti = fast_metric(
    gt,
    test_prediction
)


# ============================================================
# STATISTICS
# ============================================================

tp = int(
    np.sum(
        gt[
            sy,
            sx
        ] > 0
    )
)

selected_count = len(
    selected
)

fp = selected_count - tp

precision = (
    tp / selected_count
)

recall = (
    tp / GT_POSITIVES
)

threshold = val_prediction[
    selected[-1]
]


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 75)
print("LOCKED V1_19 RESULT")
print("=" * 75)

print(
    f"K          : {K:,}"
)

print(
    f"Power      : {POWER}"
)

print(
    f"Radius     : {RADIUS}"
)

print(
    f"Selected   : {selected_count:,}"
)

print(
    f"TP         : {tp:,}"
)

print(
    f"FP         : {fp:,}"
)

print(
    f"Precision  : {precision:.6f}"
)

print(
    f"Recall     : {recall:.6f}"
)

print(
    f"Threshold  : {threshold:.9f}"
)

print(
    f"Exact DTI  : {dti:.9f}"
)

print()

print("=" * 75)
print("EXPECTED")
print("=" * 75)

print(
    "Expected DTI : 0.243827623"
)

print("=" * 75)


# ============================================================
# FINAL MAP
# ============================================================

OUTPUT_PATH = (
    "outputs/final_test/"
    "reduced_feature_models/"
    "postprocessing_v3/"
    "V1_19_FINAL_VALIDATION_MAP.npy"
)

np.save(
    OUTPUT_PATH,
    test_prediction
)

print()
print(
    "Saved final validation map:"
)

print(
    OUTPUT_PATH
)
