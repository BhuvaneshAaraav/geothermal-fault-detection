import os
import numpy as np
import rasterio


# ============================================================
# CONFIG
# ============================================================

PREDICTION_FILE = (
    "outputs/validation_full_raster_predictions.npy"
)

LABEL_RASTER = (
    "data/raw/Training_fault_labels.tif"
)

OUTPUT_DIR = "outputs"

BEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validation_calibrated_best.npy"
)

RADIUS = 3

ALPHA = 0.2
BETA = 0.8


# ============================================================
# VALIDATION BLOCKS
# ============================================================

VALIDATION_BLOCKS = {
    1,
    7,
    9,
    10
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("LOADING FULL-RASTER PREDICTIONS")
print("=" * 60)

prediction = np.load(
    PREDICTION_FILE
).astype(np.float32)

height, width = prediction.shape

print(
    "Prediction shape:",
    prediction.shape
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
    "Label shape:",
    labels.shape
)


# ============================================================
# CREATE VALIDATION MASK
# ============================================================

rows_per_block = height // 4
cols_per_block = width // 4

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

    if block_id in VALIDATION_BLOCKS:

        validation_mask[
            y0:y1,
            x0:x1
        ] = True


# ============================================================
# VALIDATION DATA ONLY
# ============================================================

p_original = np.where(
    validation_mask,
    prediction,
    0.0
).astype(np.float32)

g = np.where(
    validation_mask,
    labels,
    0.0
).astype(np.float32)


# ============================================================
# ZERO-PADDING SHIFT
# ============================================================

def shift_zero(
    array,
    dy,
    dx
):

    h, w = array.shape

    result = np.zeros_like(
        array,
        dtype=np.float32
    )

    src_y0 = max(
        0,
        dy
    )

    src_y1 = min(
        h,
        h + dy
    )

    dst_y0 = max(
        0,
        -dy
    )

    dst_y1 = (
        dst_y0
        + src_y1
        - src_y0
    )

    src_x0 = max(
        0,
        dx
    )

    src_x1 = min(
        w,
        w + dx
    )

    dst_x0 = max(
        0,
        -dx
    )

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
# PRECOMPUTE GT INFLUENCE
#
# This does not depend on the probability calibration,
# so calculate it only once.
# ============================================================

print()
print("=" * 60)
print("CALCULATING GT SPATIAL INFLUENCE")
print("=" * 60)

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

        shifted_gt = (
            shift_zero(
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


# ============================================================
# TVERSKY FUNCTION
# ============================================================

def calculate_tversky(p):

    # --------------------------------------------------------
    # TP matching
    # --------------------------------------------------------

    tp_match = np.zeros_like(
        p,
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
                shift_zero(
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

    # --------------------------------------------------------
    # Metric components
    # --------------------------------------------------------

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

    if denominator <= 0:

        score = 0.0

    else:

        score = (
            tp / denominator
        )

    return (
        score,
        tp,
        fp,
        fn
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 60)
print("BASELINE")
print("=" * 60)

score, tp, fp, fn = calculate_tversky(
    p_original
)

print(
    f"Power 1.00"
)

print(
    f"TP = {tp:.4f}"
)

print(
    f"FP = {fp:.4f}"
)

print(
    f"FN = {fn:.4f}"
)

print(
    f"Tversky = {score:.6f}"
)


# ============================================================
# CALIBRATION POWERS
# ============================================================

powers = [
    1.00,
    1.10,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00
]


results = []


print()
print("=" * 60)
print("CALIBRATION SWEEP")
print("=" * 60)

for power in powers:

    print(
        f"\nTesting p^{power:.2f} ..."
    )

    calibrated = np.power(
        p_original,
        power
    ).astype(
        np.float32
    )

    score, tp, fp, fn = (
        calculate_tversky(
            calibrated
        )
    )

    results.append(
        (
            power,
            score,
            tp,
            fp,
            fn
        )
    )

    print(
        f"TP      : {tp:.4f}"
    )

    print(
        f"FP      : {fp:.4f}"
    )

    print(
        f"FN      : {fn:.4f}"
    )

    print(
        f"Tversky : {score:.6f}"
    )


# ============================================================
# FIND BEST
# ============================================================

best = max(
    results,
    key=lambda x: x[1]
)

best_power = best[0]
best_score = best[1]
best_tp = best[2]
best_fp = best[3]
best_fn = best[4]


# ============================================================
# SAVE BEST CALIBRATED MAP
# ============================================================

best_prediction = np.power(
    p_original,
    best_power
).astype(
    np.float32
)

np.save(
    BEST_OUTPUT,
    best_prediction
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)

print(
    f"{'Power':>8} "
    f"{'Tversky':>12} "
    f"{'TP':>14} "
    f"{'FP':>14} "
    f"{'FN':>14}"
)

print("-" * 68)

for power, score, tp, fp, fn in results:

    print(
        f"{power:8.2f} "
        f"{score:12.6f} "
        f"{tp:14.3f} "
        f"{fp:14.3f} "
        f"{fn:14.3f}"
    )


print()
print("=" * 60)
print("BEST CALIBRATION")
print("=" * 60)

print(
    f"Best power : {best_power:.2f}"
)

print(
    f"Best TP    : {best_tp:.4f}"
)

print(
    f"Best FP    : {best_fp:.4f}"
)

print(
    f"Best FN    : {best_fn:.4f}"
)

print(
    f"Best Tversky : {best_score:.6f}"
)

print()
print(
    "Saved best calibrated prediction:"
)

print(
    BEST_OUTPUT
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)