import os
import numpy as np
import rasterio


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_DIR = "outputs/v1_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RADIUS = 3
ALPHA = 0.2
BETA = 0.8
EPS = 1e-12


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING V1 VALIDATION PREDICTIONS")
print("=" * 70)

pred = np.load(PRED_PATH).astype(np.float64)

with rasterio.open(LABELS_PATH) as src:
    labels = src.read(1)

with rasterio.open(FEATURES_PATH) as src:
    valid_mask = src.read_masks(1) > 0

print("Prediction shape:", pred.shape)
print("Label shape:", labels.shape)
print("Valid feature pixels:", valid_mask.sum())


# ============================================================
# VALIDATION BLOCKS
# ============================================================

height, width = labels.shape

n_blocks_y = 4
n_blocks_x = 4

block_h = height // n_blocks_y
block_w = width // n_blocks_x

validation_blocks = [1, 7, 9, 10]

validation_mask = np.zeros_like(labels, dtype=bool)

for block_id in validation_blocks:

    by = block_id // n_blocks_x
    bx = block_id % n_blocks_x

    y0 = by * block_h
    y1 = (by + 1) * block_h if by < n_blocks_y - 1 else height

    x0 = bx * block_w
    x1 = (bx + 1) * block_w if bx < n_blocks_x - 1 else width

    validation_mask[y0:y1, x0:x1] = True


# Only evaluate valid raster pixels
evaluation_mask = validation_mask & valid_mask

y_true = (labels > 0) & evaluation_mask
y_pred = pred.copy()

y_pred[~evaluation_mask] = 0.0

print()
print("=" * 70)
print("VALIDATION DATA")
print("=" * 70)

print("Validation pixels:", validation_mask.sum())
print("Valid evaluation pixels:", evaluation_mask.sum())
print("Validation fault pixels:", y_true.sum())
print(
    "Fault percentage:",
    100.0 * y_true.sum() / evaluation_mask.sum()
)


# ============================================================
# BASIC PREDICTION STATISTICS
# ============================================================

background = y_pred[evaluation_mask & ~y_true]
fault = y_pred[y_true]

print()
print("=" * 70)
print("PREDICTION DISTRIBUTIONS")
print("=" * 70)

print("\nTRUE FAULT PROBABILITIES")
print("--------------------------------")

if len(fault) > 0:
    print("Count :", len(fault))
    print("Min   :", fault.min())
    print("P01   :", np.percentile(fault, 1))
    print("P05   :", np.percentile(fault, 5))
    print("P10   :", np.percentile(fault, 10))
    print("P25   :", np.percentile(fault, 25))
    print("Median:", np.median(fault))
    print("P75   :", np.percentile(fault, 75))
    print("P90   :", np.percentile(fault, 90))
    print("P95   :", np.percentile(fault, 95))
    print("P99   :", np.percentile(fault, 99))
    print("Max   :", fault.max())


print("\nBACKGROUND PROBABILITIES")
print("--------------------------------")

print("Count :", len(background))
print("Min   :", background.min())
print("P01   :", np.percentile(background, 1))
print("P05   :", np.percentile(background, 5))
print("P10   :", np.percentile(background, 10))
print("P25   :", np.percentile(background, 25))
print("Median:", np.median(background))
print("P75   :", np.percentile(background, 75))
print("P90   :", np.percentile(background, 90))
print("P95   :", np.percentile(background, 95))
print("P99   :", np.percentile(background, 99))
print("Max   :", background.max())


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

print(
    f"{'Threshold':>10} "
    f"{'Predicted':>12} "
    f"{'TP':>10} "
    f"{'FP':>12} "
    f"{'FN':>10} "
    f"{'Recall':>10} "
    f"{'Precision':>10}"
)

print("-" * 80)

for t in thresholds:

    positive_prediction = (y_pred >= t) & evaluation_mask

    tp = np.sum(positive_prediction & y_true)
    fp = np.sum(positive_prediction & ~y_true)
    fn = np.sum(~positive_prediction & y_true)

    recall = tp / (tp + fn + EPS)
    precision = tp / (tp + fp + EPS)

    print(
        f"{t:10.3f} "
        f"{positive_prediction.sum():12d} "
        f"{tp:10d} "
        f"{fp:12d} "
        f"{fn:10d} "
        f"{recall:10.4f} "
        f"{precision:10.4f}"
    )


# ============================================================
# VERIFIED FAST DISTANCE-WEIGHTED TVERSKY
# ============================================================

def fast_distance_tversky(prediction, ground_truth, radius=3,
                           alpha=0.2, beta=0.8):

    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=bool)

    h, w = prediction.shape

    gt_y, gt_x = np.where(ground_truth)

    if len(gt_y) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # --------------------------------------------------------
    # Maximum weighted prediction around each GT pixel
    # --------------------------------------------------------

    tp_best = np.zeros(len(gt_y), dtype=np.float64)

    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            distance = np.sqrt(dx * dx + dy * dy)

            if distance > radius:
                continue

            weight = max(1.0 - distance / radius, 0.0)

            yy = gt_y + dy
            xx = gt_x + dx

            inside = (
                (yy >= 0) &
                (yy < h) &
                (xx >= 0) &
                (xx < w)
            )

            values = np.zeros(len(gt_y), dtype=np.float64)

            values[inside] = prediction[
                yy[inside],
                xx[inside]
            ] * weight

            tp_best = np.maximum(tp_best, values)

    tp = tp_best.sum()
    fn = np.sum(1.0 - tp_best)

    # --------------------------------------------------------
    # Maximum GT proximity around each prediction pixel
    # --------------------------------------------------------

    gt_float = ground_truth.astype(np.float64)

    gt_proximity = np.zeros_like(prediction, dtype=np.float64)

    for dy in range(-radius, radius + 1):

        for dx in range(-radius, radius + 1):

            distance = np.sqrt(dx * dx + dy * dy)

            if distance > radius:
                continue

            weight = max(1.0 - distance / radius, 0.0)

            shifted = np.zeros_like(gt_float)

            y_src0 = max(0, -dy)
            y_src1 = min(h, h - dy)

            x_src0 = max(0, -dx)
            x_src1 = min(w, w - dx)

            y_dst0 = max(0, dy)
            y_dst1 = min(h, h + dy)

            x_dst0 = max(0, dx)
            x_dst1 = min(w, w + dx)

            shifted[
                y_dst0:y_dst1,
                x_dst0:x_dst1
            ] = gt_float[
                y_src0:y_src1,
                x_src0:x_src1
            ]

            gt_proximity = np.maximum(
                gt_proximity,
                shifted * weight
            )

    fp = np.sum(
        prediction *
        (1.0 - gt_proximity)
    )

    dti = tp / (
        tp +
        alpha * fp +
        beta * fn +
        EPS
    )

    return dti, tp, fp, fn


# ============================================================
# DTI FOR RAW V1
# ============================================================

print()
print("=" * 70)
print("RAW V1 DISTANCE-WEIGHTED TVERSKY")
print("=" * 70)

raw_dti, raw_tp, raw_fp, raw_fn = fast_distance_tversky(
    y_pred,
    y_true,
    radius=RADIUS,
    alpha=ALPHA,
    beta=BETA
)

print("DTI:", raw_dti)
print("Weighted TP:", raw_tp)
print("Weighted FP:", raw_fp)
print("Weighted FN:", raw_fn)


# ============================================================
# POWER ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PROBABILITY POWER ANALYSIS")
print("=" * 70)

powers = [
    0.75,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
]

for power in powers:

    transformed = np.zeros_like(y_pred)

    valid = evaluation_mask

    transformed[valid] = np.power(
        np.clip(y_pred[valid], 0.0, 1.0),
        power
    )

    dti, tp, fp, fn = fast_distance_tversky(
        transformed,
        y_true,
        radius=RADIUS,
        alpha=ALPHA,
        beta=BETA
    )

    print(
        f"power={power:5.2f} "
        f"DTI={dti:.6f} "
        f"TP={tp:.2f} "
        f"FP={fp:.2f} "
        f"FN={fn:.2f}"
    )


# ============================================================
# POWER + THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("POWER + THRESHOLD ANALYSIS")
print("=" * 70)

powers = [1.0, 1.2, 1.5, 2.0]

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

best = None

for power in powers:

    transformed = np.zeros_like(y_pred)

    valid = evaluation_mask

    transformed[valid] = np.power(
        np.clip(y_pred[valid], 0.0, 1.0),
        power
    )

    for threshold in thresholds:

        calibrated = transformed.copy()

        calibrated[
            calibrated < threshold
        ] = 0.0

        dti, tp, fp, fn = fast_distance_tversky(
            calibrated,
            y_true,
            radius=RADIUS,
            alpha=ALPHA,
            beta=BETA
        )

        print(
            f"power={power:4.1f} "
            f"threshold={threshold:.2f} "
            f"DTI={dti:.6f}"
        )

        if best is None or dti > best[0]:

            best = (
                dti,
                power,
                threshold,
                tp,
                fp,
                fn
            )


# ============================================================
# BEST COMBINATION
# ============================================================

print()
print("=" * 70)
print("BEST POWER + THRESHOLD")
print("=" * 70)

dti, power, threshold, tp, fp, fn = best

print("DTI       :", dti)
print("Power     :", power)
print("Threshold :", threshold)
print("Weighted TP:", tp)
print("Weighted FP:", fp)
print("Weighted FN:", fn)


# ============================================================
# HIGH CONFIDENCE FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("HIGH-CONFIDENCE FALSE POSITIVES")
print("=" * 70)

for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:

    fp_mask = (
        (y_pred >= threshold) &
        evaluation_mask &
        ~y_true
    )

    ys, xs = np.where(fp_mask)

    print(
        f"Probability >= {threshold:.2f}: "
        f"{len(xs):,} false-positive pixels"
    )

    if len(xs) > 0:

        scores = y_pred[ys, xs]

        order = np.argsort(scores)[::-1]

        print("Top 10 locations:")

        for idx in order[:10]:

            print(
                f"  row={ys[idx]:4d} "
                f"col={xs[idx]:4d} "
                f"prob={scores[idx]:.6f}"
            )


# ============================================================
# SAVE DISTRIBUTIONS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "fault_probabilities.npy"
    ),
    fault
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "background_probabilities.npy"
    ),
    background
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved distributions to:",
    OUTPUT_DIR
)
