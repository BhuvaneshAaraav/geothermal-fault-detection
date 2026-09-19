import numpy as np
import rasterio
from src19_metric import fast_metric

# ============================================================
# V1 SPATIAL + K SEARCH
# ============================================================
#
# IMPORTANT:
# This script uses the SAME CANONICAL validation ground truth
# used for the V1 benchmark.
#
# DO NOT replace gt with the full label raster.
#
# Canonical evaluation:
#   gt = zeros(H, W)
#   gt[validation_coords] = validation labels
#
# This keeps results directly comparable with:
#
# V1 raw                  = 0.148330184
# V1 Top-K + power       = 0.206570046
# V1 + 100m thinning     = 0.230572555
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

PRED_PATH = (
    "outputs/validation_probability_map.npy"
)

OUTPUT_PATH = (
    "outputs/v1_spatial_k_best.npy"
)


# ============================================================
# FIXED PARAMETERS
# ============================================================

# Winning spatial distance from Script 48
#
# 1 raster pixel = 100 meters
#
RADIUS = 1

# Winning V1 power
POWER = 0.005


# ============================================================
# K SEARCH
# ============================================================

K_VALUES = [
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    80000,
    90000,
    100000,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V1 SPATIAL + K SEARCH")
print("=" * 70)

print("\nLoading labels...")

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

H, W = labels.shape

print("Label shape:", labels.shape)


print("\nLoading validation coordinates...")

val_coords = np.load(
    VAL_COORDS_PATH
)

print(
    "Validation coordinates:",
    val_coords.shape
)


print("\nLoading V1 predictions...")

prediction = np.load(
    PRED_PATH
).astype(np.float64)

print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# VALIDATION COORDINATES
# ============================================================

ys = val_coords[:, 0]
xs = val_coords[:, 1]


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_prediction = prediction[
    ys,
    xs
]


# ============================================================
# VALIDATION GROUND TRUTH
# ============================================================

val_gt = (
    labels[ys, xs] > 0
)


# ============================================================
# CANONICAL 2D GROUND TRUTH
# ============================================================
#
# THIS IS CRITICAL.
#
# Only validation coordinates are written into gt.
#
# Pixels outside val_coords remain zero.
#
# This is the same protocol used for the
# V1 benchmark of 0.206570046.
#
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[
    ys,
    xs
] = val_gt.astype(np.float64)


# ============================================================
# INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("CANONICAL VALIDATION SET")
print("=" * 70)

print(
    "Raster size:",
    (H, W)
)

print(
    "Validation pixels:",
    len(val_coords)
)

print(
    "Validation GT positives:",
    int(val_gt.sum())
)


# ============================================================
# SORT V1 PREDICTIONS
# ============================================================

print("\nSorting V1 predictions...")

order = np.argsort(
    val_prediction
)[::-1]

print("Sorting complete.")


# ============================================================
# SPATIAL THINNING FUNCTION
# ============================================================

def spatial_thin(
    candidate_indices,
    max_points,
    radius
):

    # --------------------------------------------------------
    # No thinning
    # --------------------------------------------------------

    if radius == 0:

        return candidate_indices[
            :max_points
        ]


    # --------------------------------------------------------
    # Occupied map
    # --------------------------------------------------------

    occupied = np.zeros(
        (H, W),
        dtype=np.uint8
    )


    selected = []


    # --------------------------------------------------------
    # Greedy spatial selection
    # --------------------------------------------------------

    for idx in candidate_indices:

        y = ys[idx]
        x = xs[idx]


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


        # ----------------------------------------------------
        # If another selected point already occupies this
        # neighborhood, skip this prediction.
        # ----------------------------------------------------

        if np.any(
            occupied[
                y0:y1,
                x0:x1
            ]
        ):

            continue


        # ----------------------------------------------------
        # Select point
        # ----------------------------------------------------

        selected.append(
            idx
        )


        # ----------------------------------------------------
        # Mark neighborhood as occupied
        # ----------------------------------------------------

        occupied[
            y0:y1,
            x0:x1
        ] = 1


        # ----------------------------------------------------
        # Stop when K reached
        # ----------------------------------------------------

        if len(selected) >= max_points:

            break


    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# SEARCH
# ============================================================

results = []


best_score = -1.0
best_k = None
best_n = None
best_tp = None
best_precision = None
best_recall = None
best_threshold = None


print("\n")
print("=" * 70)
print("SPATIAL K SEARCH")
print("=" * 70)

print(
    f"Fixed radius : {RADIUS} pixel (~{RADIUS * 100} m)"
)

print(
    f"Fixed power  : {POWER}"
)


# ============================================================
# LOOP OVER K
# ============================================================

for k in K_VALUES:

    print("\n")
    print("-" * 70)

    print(
        f"Testing K = {k:,}"
    )


    # --------------------------------------------------------
    # Spatially thin the ranked predictions
    # --------------------------------------------------------

    selected = spatial_thin(
        order,
        k,
        RADIUS
    )


    n_selected = len(
        selected
    )


    if n_selected == 0:

        print(
            "No predictions selected."
        )

        continue


    # --------------------------------------------------------
    # Selected coordinates
    # --------------------------------------------------------

    selected_y = ys[
        selected
    ]

    selected_x = xs[
        selected
    ]


    # --------------------------------------------------------
    # Selected V1 scores
    # --------------------------------------------------------

    selected_values = (
        val_prediction[selected]
    )


    threshold = (
        selected_values[-1]
    )


    # --------------------------------------------------------
    # Power transformation
    # --------------------------------------------------------

    transformed = np.power(
        np.clip(
            selected_values,
            0.0,
            1.0
        ),
        POWER
    )


    # --------------------------------------------------------
    # Full 2D prediction map
    # --------------------------------------------------------

    test_prediction = np.zeros(
        (H, W),
        dtype=np.float64
    )


    test_prediction[
        selected_y,
        selected_x
    ] = transformed


    # --------------------------------------------------------
    # EXACT VERIFIED COMPETITION METRIC
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # fast_metric(gt, test_prediction)
    #
    # NOT:
    #
    # fast_metric(test_prediction, gt)
    #
    # AND NOT the full labels raster.
    #
    # --------------------------------------------------------

    score = fast_metric(
        gt,
        test_prediction
    )


    # --------------------------------------------------------
    # Classification statistics
    # --------------------------------------------------------

    tp = int(
        np.sum(
            val_gt[selected]
        )
    )


    precision = (
        tp / n_selected
    )


    recall = (
        tp / np.sum(val_gt)
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        (
            score,
            k,
            n_selected,
            tp,
            precision,
            recall,
            threshold
        )
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Selected : {n_selected}"
    )

    print(
        f"TP       : {tp}"
    )

    print(
        f"FP       : {n_selected - tp}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"Threshold: {threshold:.9f}"
    )

    print(
        f"DTI      : {score:.9f}"
    )


    # --------------------------------------------------------
    # Save new best
    # --------------------------------------------------------

    if score > best_score:

        best_score = score

        best_k = k

        best_n = n_selected

        best_tp = tp

        best_precision = precision

        best_recall = recall

        best_threshold = threshold


        np.save(
            OUTPUT_PATH,
            test_prediction
        )


        print(
            "\n🔥 NEW BEST"
        )


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

print("\n")
print("=" * 70)
print("TOP RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'K':<10}"
    f"{'Points':<10}"
    f"{'TP':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'DTI':<14}"
)


for rank, result in enumerate(
    results,
    start=1
):

    (
        score,
        k,
        n,
        tp,
        precision,
        recall,
        threshold
    ) = result


    print(
        f"{rank:<6}"
        f"{k:<10}"
        f"{n:<10}"
        f"{tp:<8}"
        f"{precision:<12.6f}"
        f"{recall:<12.6f}"
        f"{score:<14.9f}"
    )


# ============================================================
# BEST RESULT
# ============================================================

print("\n")
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(
    f"Radius       : "
    f"{RADIUS} pixel (~{RADIUS * 100} m)"
)

print(
    f"Power        : {POWER}"
)

print(
    f"Best K       : {best_k:,}"
)

print(
    f"Selected     : {best_n:,}"
)

print(
    f"TP           : {best_tp:,}"
)

print(
    f"FP           : {best_n - best_tp:,}"
)

print(
    f"Precision    : {best_precision:.6f}"
)

print(
    f"Recall       : {best_recall:.6f}"
)

print(
    f"Threshold    : {best_threshold:.9f}"
)

print(
    f"DTI          : {best_score:.9f}"
)


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

ORIGINAL_V1 = 0.206570046
SPATIAL_V1 = 0.230572555


print("\n")
print("=" * 70)
print("BENCHMARK COMPARISON")
print("=" * 70)

print(
    f"V1 raw                 : "
    f"{0.148330184:.9f}"
)

print(
    f"V1 Top-K + power       : "
    f"{ORIGINAL_V1:.9f}"
)

print(
    f"V1 + 100m thinning     : "
    f"{SPATIAL_V1:.9f}"
)

print(
    f"V1 + optimized K       : "
    f"{best_score:.9f}"
)


print(
    f"\nImprovement over "
    f"original V1: "
    f"{best_score - ORIGINAL_V1:+.9f}"
)

print(
    f"Improvement over "
    f"100m result: "
    f"{best_score - SPATIAL_V1:+.9f}"
)


# ============================================================
# SAVE BEST CONFIGURATION
# ============================================================

config_path = (
    "outputs/v1_spatial_k_best_config.txt"
)

with open(
    config_path,
    "w"
) as f:

    f.write(
        "V1 Spatial K Search Best Configuration\n"
    )

    f.write(
        "=======================================\n"
    )

    f.write(
        f"DTI={best_score:.12f}\n"
    )

    f.write(
        f"K={best_k}\n"
    )

    f.write(
        f"RADIUS={RADIUS}\n"
    )

    f.write(
        f"POWER={POWER}\n"
    )

    f.write(
        f"SELECTED={best_n}\n"
    )

    f.write(
        f"TP={best_tp}\n"
    )

    f.write(
        f"FP={best_n - best_tp}\n"
    )

    f.write(
        f"PRECISION={best_precision:.12f}\n"
    )

    f.write(
        f"RECALL={best_recall:.12f}\n"
    )

    f.write(
        f"THRESHOLD={best_threshold:.12f}\n"
    )


print(
    f"\nSaved prediction: "
    f"{OUTPUT_PATH}"
)

print(
    f"Saved configuration: "
    f"{config_path}"
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)

if best_score > SPATIAL_V1:

    print(
        "🔥 OPTIMIZED K BEAT THE PREVIOUS "
        "100m SPATIAL RESULT."
    )

elif best_score > ORIGINAL_V1:

    print(
        "🔥 SPATIAL METHOD STILL BEATS "
        "ORIGINAL V1."
    )

else:

    print(
        "No improvement over the "
        "original V1 benchmark."
    )

print("=" * 70)