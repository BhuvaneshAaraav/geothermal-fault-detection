import numpy as np
import rasterio

from src19_metric import fast_metric


LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_PATH = "outputs/v1_final_topk_best.npy"

POWER = 0.005

K_VALUES = range(
    43000,
    46001,
    25
)


print("=" * 70)
print("V1 FINAL FINE TOP-K SEARCH")
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

print(
    "\nEvaluation pixels:",
    len(val_prediction)
)

print(
    "Fault pixels:",
    int(gt.sum())
)

print(
    "\nPower:",
    POWER
)

print(
    "K range: 43,000 → 46,000"
)

print(
    "K step: 25"
)

print(
    "Total evaluations:",
    len(list(K_VALUES))
)


# ============================================================
# SORT
# ============================================================

print("\nSorting predictions...")

order = np.argsort(
    val_prediction
)[::-1]


# ============================================================
# SEARCH
# ============================================================

best_score = -1.0
best_k = None
best_threshold = None
best_prediction = None

results = []

counter = 0

for k in K_VALUES:

    counter += 1

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

    transformed = np.power(
        np.clip(
            selected_values,
            0.0,
            1.0
        ),
        POWER
    )

    test_prediction = np.zeros(
        (H, W),
        dtype=np.float64
    )

    test_prediction[
        selected_y,
        selected_x
    ] = transformed

    score = fast_metric(
        gt,
        test_prediction
    )

    tp_pixels = np.sum(
        gt[
            selected_y,
            selected_x
        ] > 0
    )

    precision = tp_pixels / k
    recall = tp_pixels / gt.sum()

    results.append(
        (
            score,
            k,
            threshold,
            int(tp_pixels),
            precision,
            recall
        )
    )

    if score > best_score:

        best_score = score
        best_k = k
        best_threshold = threshold
        best_prediction = test_prediction.copy()

        print(
            f"NEW BEST | "
            f"K={k:,} | "
            f"Threshold={threshold:.6f} | "
            f"TP={int(tp_pixels):,} | "
            f"DTI={score:.9f}"
        )


# ============================================================
# TOP 20
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print("\n" + "=" * 70)
print("TOP 20")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'DTI':<14}"
    f"{'K':<9}"
    f"{'Threshold':<13}"
    f"{'TP':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
)

for i, r in enumerate(
    results[:20],
    1
):

    (
        score,
        k,
        threshold,
        tp,
        precision,
        recall
    ) = r

    print(
        f"{i:<6}"
        f"{score:<14.9f}"
        f"{k:<9,}"
        f"{threshold:<13.6f}"
        f"{tp:<8,}"
        f"{precision:<12.5f}"
        f"{recall:<10.5f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("V1 FINAL DEVELOPMENT RESULT")
print("=" * 70)

print(
    f"\nRaw V1 DTI       : 0.148330184"
)

print(
    f"Previous best    : 0.206492796"
)

print(
    f"Best DTI         : {best_score:.9f}"
)

print(
    f"Best K           : {best_k:,}"
)

print(
    f"Power            : {POWER}"
)

print(
    f"Threshold        : {best_threshold:.6f}"
)

print(
    f"Improvement      : "
    f"{best_score - 0.148330184:+.9f}"
)

print(
    f"Relative         : "
    f"{((best_score / 0.148330184) - 1) * 100:+.2f}%"
)


np.save(
    OUTPUT_PATH,
    best_prediction
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)

print("\nDONE")