import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/v6_validation_probability.npy"

OUTPUT_PATH = "outputs/v6_fine_calibrated_best.npy"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V6 FINE CALIBRATION")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

val_coords = np.load(VAL_COORDS_PATH)

prediction = np.load(
    PRED_PATH
).astype(np.float64)

H, W = labels.shape


# ============================================================
# BUILD VALIDATION ARRAYS
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

mask = np.zeros(
    (H, W),
    dtype=bool
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

mask[ys, xs] = True

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

prediction[~mask] = 0.0
gt[~mask] = 0.0


print("\nEvaluation pixels:", int(mask.sum()))
print("Fault pixels:", int(gt.sum()))


# ============================================================
# BASELINE
# ============================================================

baseline = fast_metric(
    gt,
    prediction
)

print(f"\nRaw V6 DTI: {baseline:.9f}")


# ============================================================
# SEARCH SPACE
# ============================================================

powers = np.arange(
    0.55,
    1.501,
    0.05
)

thresholds = np.arange(
    0.85,
    0.991,
    0.01
)


print("\nPowers:", len(powers))
print("Thresholds:", len(thresholds))
print(
    "Total evaluations:",
    len(powers) * len(thresholds)
)


# ============================================================
# SEARCH
# ============================================================

best_score = baseline
best_power = 1.0
best_threshold = 0.0
best_prediction = prediction.copy()

results = []

counter = 0
total = len(powers) * len(thresholds)


for power in powers:

    transformed = np.power(
        np.clip(prediction, 0.0, 1.0),
        power
    )

    for threshold in thresholds:

        counter += 1

        calibrated = transformed.copy()

        calibrated[
            calibrated < threshold
        ] = 0.0

        score = fast_metric(
            gt,
            calibrated
        )

        count = int(
            np.count_nonzero(calibrated)
        )

        results.append(
            (
                score,
                power,
                threshold,
                count
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
                f"Predicted={count:,}"
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
print("TOP 15 RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'DTI':<14}"
    f"{'Power':<10}"
    f"{'Threshold':<12}"
    f"{'Predicted':<15}"
)

for rank, result in enumerate(
    results[:15],
    1
):

    score, power, threshold, count = result

    print(
        f"{rank:<6}"
        f"{score:<14.9f}"
        f"{power:<10.2f}"
        f"{threshold:<12.2f}"
        f"{count:<15,}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BEST V6 FINE CALIBRATION")
print("=" * 70)

print(
    f"\nRaw V6 DTI       : {baseline:.9f}"
)

print(
    f"Best DTI         : {best_score:.9f}"
)

print(
    f"Power            : {best_power:.2f}"
)

print(
    f"Threshold        : {best_threshold:.2f}"
)

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

print(
    f"\nSaved: {OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)