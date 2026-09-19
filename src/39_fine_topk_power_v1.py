import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_PATH = "outputs/v1_fine_topk_power_best.npy"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V1 FINE TOP-K + LOW-POWER SEARCH")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

val_coords = np.load(
    VAL_COORDS_PATH
)

prediction = np.load(
    PRED_PATH
).astype(np.float64)

H, W = labels.shape


# ============================================================
# VALIDATION COORDINATES
# ============================================================

ys = val_coords[:, 0]
xs = val_coords[:, 1]

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

val_prediction = prediction[
    ys,
    xs
]


print("\nEvaluation pixels:", len(val_prediction))
print("Fault pixels:", int(gt.sum()))


# ============================================================
# SORT ONCE
# ============================================================

print("\nSorting predictions...")

order = np.argsort(
    val_prediction
)[::-1]


# ============================================================
# SEARCH SPACE
# ============================================================

K_VALUES = range(
    35000,
    55001,
    1000
)

POWERS = np.arange(
    0.05,
    0.301,
    0.01
)


print("\nK range:")
print("35,000 → 55,000")

print("\nPower range:")
print("0.05 → 0.30")

total = len(list(K_VALUES)) * len(POWERS)

print(
    "\nTotal evaluations:",
    total
)


# ============================================================
# SEARCH
# ============================================================

best_score = -1.0
best_k = None
best_power = None
best_threshold = None
best_prediction = None

results = []

counter = 0


for k in K_VALUES:

    selected = order[:k]

    selected_values = val_prediction[
        selected
    ]

    selected_y = ys[
        selected
    ]

    selected_x = xs[
        selected
    ]

    threshold = selected_values[-1]

    for power in POWERS:

        counter += 1

        # ----------------------------------------------------
        # Power transformation
        # ----------------------------------------------------

        transformed = np.power(
            np.clip(
                selected_values,
                0.0,
                1.0
            ),
            power
        )

        # ----------------------------------------------------
        # Full prediction map
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
        # EXACT COMPETITION DTI
        # ----------------------------------------------------

        score = fast_metric(
            gt,
            test_prediction
        )

        # ----------------------------------------------------
        # Pixel diagnostics
        # ----------------------------------------------------

        tp_pixels = np.sum(
            gt[
                selected_y,
                selected_x
            ] > 0
        )

        precision = (
            tp_pixels / k
        )

        recall = (
            tp_pixels / gt.sum()
        )

        results.append(
            (
                score,
                k,
                power,
                threshold,
                int(tp_pixels),
                precision,
                recall
            )
        )

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        if score > best_score:

            best_score = score
            best_k = k
            best_power = power
            best_threshold = threshold
            best_prediction = (
                test_prediction.copy()
            )

            print(
                f"NEW BEST | "
                f"K={k:,} | "
                f"Power={power:.2f} | "
                f"Threshold={threshold:.6f} | "
                f"TP={int(tp_pixels):,} | "
                f"DTI={score:.9f}"
            )

        if counter % 50 == 0:

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
print("TOP 20 RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'DTI':<14}"
    f"{'K':<9}"
    f"{'Power':<9}"
    f"{'Threshold':<14}"
    f"{'TP':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
)

for rank, result in enumerate(
    results[:20],
    1
):

    (
        score,
        k,
        power,
        threshold,
        tp,
        precision,
        recall
    ) = result

    print(
        f"{rank:<6}"
        f"{score:<14.9f}"
        f"{k:<9,}"
        f"{power:<9.2f}"
        f"{threshold:<14.6f}"
        f"{tp:<8,}"
        f"{precision:<12.5f}"
        f"{recall:<10.5f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BEST FINE TOP-K + POWER RESULT")
print("=" * 70)

print(
    f"\nV1 raw DTI          : 0.148330184"
)

print(
    f"V1 calibrated DTI   : 0.200007247"
)

print(
    f"Previous best       : 0.203866470"
)

print(
    f"Best DTI            : {best_score:.9f}"
)

print(
    f"Best K              : {best_k:,}"
)

print(
    f"Best Power          : {best_power:.2f}"
)

print(
    f"Threshold           : {best_threshold:.6f}"
)

print(
    f"Predicted pixels    : {best_k:,}"
)

print(
    f"\nImprovement vs raw  : "
    f"{best_score - 0.148330184:+.9f}"
)

print(
    f"Relative improvement: "
    f"{(best_score / 0.148330184 - 1) * 100:+.2f}%"
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)