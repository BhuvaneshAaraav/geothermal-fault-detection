import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/v6_validation_probability.npy"

OUTPUT_PATH = "outputs/v6_calibrated_best.npy"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V6 CALIBRATION — EXACT COMPETITION METRIC")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

val_coords = np.load(VAL_COORDS_PATH)
pred = np.load(PRED_PATH).astype(np.float64)

H, W = labels.shape

print("\nLabels:", labels.shape)
print("Validation coordinates:", val_coords.shape)
print("Prediction:", pred.shape)


# ============================================================
# BUILD EVALUATION ARRAYS
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

for y, x in val_coords:
    gt[y, x] = labels[y, x] > 0


# Prediction is already a full raster.
prediction = pred.copy()

# Only validation region should contribute.
validation_mask = np.zeros(
    (H, W),
    dtype=bool
)

validation_mask[
    val_coords[:, 0],
    val_coords[:, 1]
] = True


# Make everything outside validation zero.
prediction[~validation_mask] = 0.0
gt[~validation_mask] = 0.0


print("\nEvaluation pixels:", int(validation_mask.sum()))
print("Fault pixels:", int(gt.sum()))

print("\nOriginal prediction:")
print("Min :", prediction[validation_mask].min())
print("Max :", prediction[validation_mask].max())
print("Mean:", prediction[validation_mask].mean())
print("Median:", np.median(prediction[validation_mask]))


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

baseline = fast_metric(
    gt,
    prediction
)

print(f"Raw V6 DTI: {baseline:.9f}")


# ============================================================
# POWER SEARCH
# ============================================================

powers = [
    0.50,
    0.70,
    0.80,
    0.90,
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
    3.50,
    4.00,
    5.00,
]


# Threshold is applied after power transformation.
thresholds = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# SEARCH
# ============================================================

best_score = baseline
best_power = 1.0
best_threshold = 0.0
best_prediction = prediction.copy()

results = []

total = len(powers) * len(thresholds)
counter = 0

print("\n" + "=" * 70)
print("SEARCHING")
print("=" * 70)

for power in powers:

    transformed = np.power(
        np.clip(prediction, 0.0, 1.0),
        power
    )

    for threshold in thresholds:

        counter += 1

        # ----------------------------------------------------
        # Threshold
        #
        # Important:
        # Keep probabilities rather than making them binary.
        # The competition metric accepts soft predictions.
        # ----------------------------------------------------

        calibrated = transformed.copy()

        calibrated[calibrated < threshold] = 0.0

        score = fast_metric(
            gt,
            calibrated
        )

        results.append(
            (
                score,
                power,
                threshold,
                int(np.count_nonzero(calibrated))
            )
        )

        if score > best_score:

            best_score = score
            best_power = power
            best_threshold = threshold
            best_prediction = calibrated.copy()

            print(
                f"NEW BEST | "
                f"Power={power:.2f} | "
                f"Threshold={threshold:.2f} | "
                f"DTI={score:.9f} | "
                f"Predicted={np.count_nonzero(calibrated):,}"
            )

        if counter % 25 == 0:

            print(
                f"Progress: {counter}/{total}"
            )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    best_prediction.astype(np.float32)
)


# ============================================================
# TOP RESULTS
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print("\n" + "=" * 70)
print("TOP 10 RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'DTI':<14}"
    f"{'Power':<10}"
    f"{'Threshold':<12}"
    f"{'Predicted':<15}"
)

for i, result in enumerate(results[:10], 1):

    score, power, threshold, count = result

    print(
        f"{i:<6}"
        f"{score:<14.9f}"
        f"{power:<10.2f}"
        f"{threshold:<12.2f}"
        f"{count:<15,}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BEST V6 CALIBRATION")
print("=" * 70)

print(f"\nRaw V6 DTI       : {baseline:.9f}")
print(f"Best calibrated  : {best_score:.9f}")
print(f"Power            : {best_power:.2f}")
print(f"Threshold        : {best_threshold:.2f}")
print(
    f"Predicted pixels : "
    f"{np.count_nonzero(best_prediction):,}"
)

print(
    f"\nImprovement      : "
    f"{best_score - baseline:+.9f}"
)

print(
    f"Relative change  : "
    f"{(best_score / baseline - 1) * 100:+.2f}%"
)

print(f"\nSaved: {OUTPUT_PATH}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
