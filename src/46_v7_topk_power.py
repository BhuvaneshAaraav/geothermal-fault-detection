import numpy as np
import rasterio
from src19_metric import fast_metric

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/v7_validation_probability.npy"

OUTPUT_PATH = "outputs/v7_topk_power_best.npy"

# ============================================================
# SEARCH RANGE
# ============================================================

K_VALUES = range(20_000, 100_001, 5_000)

POWER_VALUES = np.arange(0.005, 0.1001, 0.005)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V7 TOP-K + POWER SEARCH")
print("=" * 70)

print("\nLoading labels...")

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

print("Label shape:", labels.shape)

print("\nLoading validation coordinates...")

val_coords = np.load(VAL_COORDS_PATH)

print("Validation coordinates:", val_coords.shape)

print("\nLoading V7 predictions...")

prediction = np.load(PRED_PATH).astype(np.float64)

print("Prediction shape:", prediction.shape)

# ============================================================
# BUILD CANONICAL GROUND TRUTH
# ============================================================

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]

gt = np.zeros((H, W), dtype=np.float64)

gt[ys, xs] = (labels[ys, xs] > 0).astype(np.float64)

val_prediction = prediction[ys, xs]

print("\nCanonical validation pixels:", len(val_prediction))
print("Ground-truth positives:", int(gt.sum()))

# ============================================================
# SORT PREDICTIONS
# ============================================================

print("\nSorting predictions...")

order = np.argsort(val_prediction)[::-1]

print("Sorting complete.")

# ============================================================
# SEARCH
# ============================================================

best_score = -1.0
best_k = None
best_power = None
best_threshold = None
best_tp = None

results = []

total = len(K_VALUES) * len(POWER_VALUES)
current = 0

print("\n")
print("=" * 70)
print("SEARCHING")
print("=" * 70)

for k in K_VALUES:

    selected = order[:k]

    selected_values = val_prediction[selected]

    selected_y = ys[selected]
    selected_x = xs[selected]

    threshold = selected_values[-1]

    for power in POWER_VALUES:

        current += 1

        # ----------------------------------------------------
        # POWER TRANSFORMATION
        # ----------------------------------------------------

        transformed = np.power(
            np.clip(selected_values, 0.0, 1.0),
            power
        )

        # ----------------------------------------------------
        # BUILD FULL 2D PREDICTION MAP
        # ----------------------------------------------------

        test_prediction = np.zeros(
            (H, W),
            dtype=np.float64
        )

        test_prediction[
            selected_y,
            selected_x
        ] = transformed

        # ----------------------------------------------------
        # EXACT COMPETITION METRIC
        # ----------------------------------------------------

        score = fast_metric(
            gt,
            test_prediction
        )

        # ----------------------------------------------------
        # TP AT BINARY SELECTION LEVEL
        # ----------------------------------------------------

        tp = int(
            np.sum(
                labels[selected_y, selected_x] > 0
            )
        )

        results.append(
            (
                score,
                k,
                power,
                threshold,
                tp
            )
        )

        if score > best_score:

            best_score = score
            best_k = k
            best_power = power
            best_threshold = threshold
            best_tp = tp

            np.save(
                OUTPUT_PATH,
                test_prediction
            )

            print(
                f"\nNEW BEST"
                f" | score={best_score:.9f}"
                f" | K={best_k}"
                f" | power={best_power:.3f}"
                f" | threshold={best_threshold:.6f}"
                f" | TP={best_tp}"
            )

        if current % 20 == 0:

            print(
                f"Progress: {current}/{total}"
            )

# ============================================================
# TOP RESULTS
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print("\n")
print("=" * 70)
print("TOP 20 RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'Score':<14}"
    f"{'K':<10}"
    f"{'Power':<10}"
    f"{'Threshold':<14}"
    f"{'TP':<10}"
)

for i, result in enumerate(results[:20], start=1):

    score, k, power, threshold, tp = result

    print(
        f"{i:<6}"
        f"{score:<14.9f}"
        f"{k:<10}"
        f"{power:<10.3f}"
        f"{threshold:<14.6f}"
        f"{tp:<10}"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("V7 BEST RESULT")
print("=" * 70)

print(f"Best DTI       : {best_score:.9f}")
print(f"Best K         : {best_k}")
print(f"Best power     : {best_power:.3f}")
print(f"Threshold      : {best_threshold:.6f}")
print(f"TP             : {best_tp}")
print(f"Output         : {OUTPUT_PATH}")

print("\n")
print("=" * 70)
print("V1 REFERENCE")
print("=" * 70)

print("V1 raw DTI     : 0.148330184")
print("V1 optimized   : 0.206570046")

print("\n")
print("=" * 70)

if best_score > 0.206570046:

    print("🔥 V7 BEAT V1!")
    print("V7 deserves further optimization.")

elif best_score > 0.148330184:

    print("🟡 V7 ranking is useful, but V1 is still stronger.")
    print("We can investigate V7 further, but V1 remains the benchmark.")

else:

    print("❌ V7 did not beat V1.")
    print("Do not spend more training time on V7.")

print("=" * 70)