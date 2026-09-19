import os

import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

PRED_PATH = (
    "outputs/final_test/"
    "reduced_feature_models/"
    "tiled_inference/"
    "V1_19_HN_full_prediction.npy"
)

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

K = 81_500

POWER = 0.00005

RADIUS_VALUES = [0, 1, 2, 3]


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 75)
print("V1_19_HN RADIUS SEARCH")
print("=" * 75)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    H, W = labels.shape


# ============================================================
# LOAD CANONICAL VALIDATION COORDINATES
# ============================================================

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]

xs = val_coords[:, 1]

print(
    f"Validation coordinates: {len(val_coords):,}"
)


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

GT_POSITIVES = int(
    gt.sum()
)

print(
    f"Validation GT positives: {GT_POSITIVES:,}"
)


# ============================================================
# LOAD V1_19_HN PREDICTION
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
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# EXACT VALIDATION PREPROCESSING
# MATCHES SCRIPT 80
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
# MATCHES SCRIPT 80
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
# TEST EACH RADIUS
# ============================================================

results = []

print()

print(
    f"{'Radius':>10} "
    f"{'Selected':>10} "
    f"{'TP':>8} "
    f"{'FP':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'DTI':>14}"
)

print("-" * 85)


for radius in RADIUS_VALUES:

    selected = spatial_thin(
        scores,
        val_coords,
        K,
        radius
    )

    sy = ys[
        selected
    ]

    sx = xs[
        selected
    ]

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

    # ========================================================
    # EXACT COMPETITION METRIC
    # ========================================================

    dti = fast_metric(
        gt,
        test_prediction
    )

    # ========================================================
    # STATISTICS
    # ========================================================

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

    fp = (
        selected_count
        - tp
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

    results.append(
        {
            "radius": radius,
            "selected": selected_count,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "dti": float(dti)
        }
    )

    print(
        f"{radius:10d} "
        f"{selected_count:10,d} "
        f"{tp:8,d} "
        f"{fp:10,d} "
        f"{precision:12.6f} "
        f"{recall:12.6f} "
        f"{dti:14.9f}"
    )


# ============================================================
# FIND BEST RADIUS
# ============================================================

best = max(
    results,
    key=lambda x: x["dti"]
)


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("=" * 75)
print("BEST RADIUS RESULT")
print("=" * 75)

print(
    f"K          : {K:,}"
)

print(
    f"Power      : {POWER}"
)

print(
    f"Radius     : {best['radius']}"
)

print(
    f"Selected   : {best['selected']:,}"
)

print(
    f"TP         : {best['tp']:,}"
)

print(
    f"FP         : {best['fp']:,}"
)

print(
    f"Precision  : {best['precision']:.6f}"
)

print(
    f"Recall     : {best['recall']:.6f}"
)

print(
    f"Exact DTI  : {best['dti']:.9f}"
)

print()

print("=" * 75)
print("REFERENCE")
print("=" * 75)

print(
    "V1_19 locked DTI : 0.243827623"
)

print(
    "V1_19_HN K-search best : 0.243728313"
)

print("=" * 75)
