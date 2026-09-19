import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_PATH = "outputs/v1_topk_best.npy"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V1 TOP-K COMPETITION OPTIMIZATION")
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
# VALIDATION REGION
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


# ============================================================
# EXTRACT VALIDATION PREDICTIONS
# ============================================================

val_prediction = prediction[
    ys,
    xs
]

fault_mask = gt[
    ys,
    xs
] > 0


print("\nEvaluation pixels:", len(val_prediction))
print("Fault pixels:", int(fault_mask.sum()))

print(
    "Prediction range:",
    val_prediction.min(),
    "to",
    val_prediction.max()
)


# ============================================================
# TOP-K VALUES
# ============================================================

K_VALUES = [
    5000,
    7500,
    10000,
    12500,
    15000,
    17500,
    20000,
    25000,
    30000,
    35000,
    40000,
    42500,
    45000,
    50000,
    60000,
    75000,
    100000,
    125000,
    150000,
    200000
]


# ============================================================
# SORT ONCE
# ============================================================

print("\nSorting predictions...")

order = np.argsort(
    val_prediction
)[::-1]


# ============================================================
# SEARCH TOP-K
# ============================================================

best_score = -1.0
best_k = None
best_prediction = None

results = []


print("\n" + "=" * 70)
print("TOP-K SEARCH")
print("=" * 70)

for k in K_VALUES:

    selected_indices = order[:k]

    calibrated_values = val_prediction[
        selected_indices
    ]

    # --------------------------------------------------------
    # Build full raster prediction
    # --------------------------------------------------------

    test_prediction = np.zeros(
        (H, W),
        dtype=np.float64
    )

    selected_y = ys[
        selected_indices
    ]

    selected_x = xs[
        selected_indices
    ]

    # Keep original probabilities.
    test_prediction[
        selected_y,
        selected_x
    ] = calibrated_values

    # --------------------------------------------------------
    # Exact DTI
    # --------------------------------------------------------

    score = fast_metric(
        gt,
        test_prediction
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    threshold = calibrated_values[-1]

    tp_pixels = np.sum(
        fault_mask[
            selected_indices
        ]
    )

    precision = (
        tp_pixels / k
        if k > 0
        else 0
    )

    recall = (
        tp_pixels / fault_mask.sum()
        if fault_mask.sum() > 0
        else 0
    )

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

    print(
        f"K={k:>7,} | "
        f"Threshold={threshold:.6f} | "
        f"TP={int(tp_pixels):>6,} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"DTI={score:.9f}"
    )

    if score > best_score:

        best_score = score
        best_k = k

        best_prediction = (
            test_prediction.copy()
        )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    best_prediction.astype(np.float32)
)


# ============================================================
# RESULTS
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print("\n" + "=" * 70)
print("TOP 10 TOP-K RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'DTI':<14}"
    f"{'K':<10}"
    f"{'Threshold':<14}"
    f"{'TP':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
)

for rank, result in enumerate(
    results[:10],
    1
):

    score, k, threshold, tp, precision, recall = result

    print(
        f"{rank:<6}"
        f"{score:<14.9f}"
        f"{k:<10,}"
        f"{threshold:<14.6f}"
        f"{tp:<10,}"
        f"{precision:<12.5f}"
        f"{recall:<10.5f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BEST TOP-K RESULT")
print("=" * 70)

print(
    f"\nV1 raw DTI       : 0.148330184"
)

print(
    f"Best Top-K DTI   : {best_score:.9f}"
)

print(
    f"Best K           : {best_k:,}"
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)