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
    "validation_calibrated_fine_best.npy"
)

RADIUS = 3
ALPHA = 0.2
BETA = 0.8

VALIDATION_BLOCKS = {
    1,
    7,
    9,
    10
}


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

prediction = np.load(
    PREDICTION_FILE
).astype(np.float32)

with rasterio.open(LABEL_RASTER) as src:
    labels = src.read(
        1,
        out_dtype="uint8"
    )

labels = labels.astype(np.float32)

height, width = prediction.shape

print(
    "Prediction:",
    prediction.shape
)

print(
    "Labels:",
    labels.shape
)


# ============================================================
# VALIDATION MASK
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
# SHIFT FUNCTION
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
# PRECOMPUTE GT INFLUENCE
# ============================================================

print()
print("=" * 60)
print("PRECOMPUTING GT INFLUENCE")
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
            dx * dx + dy * dy
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
# TVERSKY
# ============================================================

def calculate_tversky(p):

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
                dx * dx + dy * dy
            )

            if distance > RADIUS:
                continue

            kernel = (
                1.0
                - distance / RADIUS
            )

            shifted_p = (
                shift_zero(
                    p,
                    dy,
                    dx
                )
                * kernel
            )

            tp_match = np.maximum(
                tp_match,
                shifted_p
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

    if denominator <= 0:
        return 0.0, tp, fp, fn

    score = tp / denominator

    return score, tp, fp, fn


# ============================================================
# SEARCH CONFIGURATION
# ============================================================

# Fine powers around the previous optimum.
powers = np.arange(
    1.50,
    2.501,
    0.05
)

# Thresholds.
#
# threshold = 0 means ordinary p^gamma.
thresholds = [
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.060,
    0.075,
    0.100,
    0.125,
    0.150
]


# ============================================================
# SEARCH
# ============================================================

print()
print("=" * 60)
print("FINE CALIBRATION SEARCH")
print("=" * 60)

results = []

best_score = -1.0
best_power = None
best_threshold = None
best_tp = None
best_fp = None
best_fn = None
best_prediction = None

total_tests = (
    len(powers)
    * len(thresholds)
)

test_number = 0


for power in powers:

    for threshold in thresholds:

        test_number += 1

        # ----------------------------------------------------
        # Transform
        #
        # Values below threshold are removed.
        # Remaining probabilities are power transformed.
        # ----------------------------------------------------

        calibrated = np.zeros_like(
            p_original,
            dtype=np.float32
        )

        active = (
            p_original >= threshold
        )

        calibrated[active] = np.power(
            p_original[active],
            power
        )

        score, tp, fp, fn = (
            calculate_tversky(
                calibrated
            )
        )

        results.append(
            (
                score,
                power,
                threshold,
                tp,
                fp,
                fn
            )
        )

        if score > best_score:

            best_score = score
            best_power = power
            best_threshold = threshold

            best_tp = tp
            best_fp = fp
            best_fn = fn

            best_prediction = (
                calibrated.copy()
            )

        print(
            f"\rTesting "
            f"{test_number}/{total_tests} "
            f"| power={power:.2f} "
            f"| threshold={threshold:.3f} "
            f"| score={score:.6f}",
            end=""
        )

print()


# ============================================================
# SORT RESULTS
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)


# ============================================================
# TOP RESULTS
# ============================================================

print()
print("=" * 60)
print("TOP 20 CALIBRATIONS")
print("=" * 60)

print(
    f"{'Rank':>5} "
    f"{'Power':>8} "
    f"{'Threshold':>11} "
    f"{'Tversky':>12} "
    f"{'TP':>12} "
    f"{'FP':>12} "
    f"{'FN':>12}"
)

print("-" * 78)

for rank, result in enumerate(
    results[:20],
    start=1
):

    score, power, threshold, tp, fp, fn = result

    print(
        f"{rank:5d} "
        f"{power:8.2f} "
        f"{threshold:11.3f} "
        f"{score:12.6f} "
        f"{tp:12.2f} "
        f"{fp:12.2f} "
        f"{fn:12.2f}"
    )


# ============================================================
# BEST
# ============================================================

print()
print("=" * 60)
print("BEST CALIBRATION")
print("=" * 60)

print(
    f"Power     : {best_power:.2f}"
)

print(
    f"Threshold : {best_threshold:.3f}"
)

print(
    f"TP        : {best_tp:.4f}"
)

print(
    f"FP        : {best_fp:.4f}"
)

print(
    f"FN        : {best_fn:.4f}"
)

print(
    f"Tversky   : {best_score:.6f}"
)


# ============================================================
# SAVE
# ============================================================

np.save(
    BEST_OUTPUT,
    best_prediction
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