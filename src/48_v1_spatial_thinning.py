import numpy as np
import rasterio
from scipy.ndimage import maximum_filter
from src19_metric import fast_metric

# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/validation_probability_map.npy"

OUTPUT_PATH = "outputs/v1_spatial_thinning_best.npy"

# ============================================================
# V1 CHAMPION
# ============================================================

K = 44675
POWER = 0.005

# Minimum distance between selected prediction points.
#
# These are raster pixels.
#
# Resolution is 100 m/pixel, so approximately:
#
# 1  -> 100 m
# 2  -> 200 m
# 3  -> 300 m
# 5  -> 500 m
# 10 -> 1 km
# 20 -> 2 km
#
MIN_DISTANCES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V1 SPATIAL THINNING EXPERIMENT")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

val_coords = np.load(VAL_COORDS_PATH)
prediction = np.load(PRED_PATH).astype(np.float64)

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]

val_prediction = prediction[ys, xs]
val_gt = labels[ys, xs] > 0

print("\nRaster:", (H, W))
print("Validation pixels:", len(val_coords))
print("GT positives:", int(val_gt.sum()))

# ============================================================
# V1 TOP-K
# ============================================================

print("\nSorting V1 predictions...")

order = np.argsort(val_prediction)[::-1]

base_selected = order[:K]

print("V1 Top-K:", K)
print(
    "V1 threshold:",
    val_prediction[base_selected[-1]]
)

# ============================================================
# EXACT V1 BASELINE
# ============================================================

print("\nCalculating exact V1 baseline...")

base_y = ys[base_selected]
base_x = xs[base_selected]

base_values = val_prediction[base_selected]

base_transformed = np.power(
    np.clip(base_values, 0, 1),
    POWER
)

base_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

base_prediction[
    base_y,
    base_x
] = base_transformed

base_score = fast_metric(
    np.where(
        np.isin(
            np.arange(H * W),
            ys * W + xs
        ).reshape(H, W),
        0,
        0
    ),
    base_prediction
)

# Rebuild GT cleanly
gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = val_gt.astype(np.float64)

# Correct exact baseline
base_score = fast_metric(
    gt,
    base_prediction
)

base_tp = int(np.sum(val_gt[base_selected]))

print("\nV1 BASELINE")
print("-" * 70)
print(f"DTI : {base_score:.9f}")
print(f"TP  : {base_tp}")
print(f"FP  : {K - base_tp}")

# ============================================================
# SPATIAL THINNING FUNCTION
# ============================================================

def spatial_thin(candidate_indices, max_points, radius):

    if radius == 0:
        return candidate_indices[:max_points]

    # Boolean occupancy map
    occupied = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    selected = []

    # A square neighborhood is deliberately used here.
    # We are testing the spatial hypothesis, not claiming
    # this is the final optimal geometry.
    size = 2 * radius + 1

    for idx in candidate_indices:

        y = ys[idx]
        x = xs[idx]

        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)

        x0 = max(0, x - radius)
        x1 = min(W, x + radius + 1)

        if np.any(
            occupied[y0:y1, x0:x1]
        ):
            continue

        selected.append(idx)

        occupied[y0:y1, x0:x1] = 1

        if len(selected) >= max_points:
            break

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# EXPERIMENT
# ============================================================

results = []

print("\n")
print("=" * 70)
print("SPATIAL THINNING SEARCH")
print("=" * 70)

for radius in MIN_DISTANCES:

    print(
        f"\nTesting minimum distance: "
        f"{radius} pixels "
        f"(~{radius * 100} m)"
    )

    selected = spatial_thin(
        order,
        K,
        radius
    )

    n_selected = len(selected)

    if n_selected == 0:
        print("No points selected.")
        continue

    selected_y = ys[selected]
    selected_x = xs[selected]

    selected_values = val_prediction[selected]

    transformed = np.power(
        np.clip(selected_values, 0, 1),
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

    tp = int(
        np.sum(
            val_gt[selected]
        )
    )

    precision = tp / n_selected
    recall = tp / np.sum(val_gt)

    threshold = selected_values[-1]

    results.append(
        (
            score,
            radius,
            n_selected,
            tp,
            precision,
            recall,
            threshold
        )
    )

    print(
        f"Selected : {n_selected}"
    )

    print(
        f"TP       : {tp}"
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

# ============================================================
# RESULTS
# ============================================================

results.sort(
    key=lambda x: x[0],
    reverse=True
)

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'Radius':<10}"
    f"{'Points':<10}"
    f"{'TP':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'DTI':<14}"
)

for rank, result in enumerate(results, 1):

    score, radius, n, tp, precision, recall, threshold = result

    print(
        f"{rank:<6}"
        f"{radius:<10}"
        f"{n:<10}"
        f"{tp:<8}"
        f"{precision:<12.6f}"
        f"{recall:<12.6f}"
        f"{score:<14.9f}"
    )

# ============================================================
# SAVE BEST
# ============================================================

best = results[0]

(
    best_score,
    best_radius,
    best_n,
    best_tp,
    best_precision,
    best_recall,
    best_threshold
) = best

selected = spatial_thin(
    order,
    K,
    best_radius
)

selected_y = ys[selected]
selected_x = xs[selected]

selected_values = val_prediction[selected]

transformed = np.power(
    np.clip(selected_values, 0, 1),
    POWER
)

best_prediction = np.zeros(
    (H, W),
    dtype=np.float64
)

best_prediction[
    selected_y,
    selected_x
] = transformed

np.save(
    OUTPUT_PATH,
    best_prediction
)

print("\n")
print("=" * 70)
print("BEST SPATIAL THINNING RESULT")
print("=" * 70)

print(f"Baseline DTI : {base_score:.9f}")
print(f"Best DTI     : {best_score:.9f}")
print(f"Radius       : {best_radius} pixels")
print(f"Distance     : ~{best_radius * 100} m")
print(f"Points       : {best_n}")
print(f"TP           : {best_tp}")
print(f"Precision    : {best_precision:.6f}")
print(f"Recall       : {best_recall:.6f}")
print(f"Threshold    : {best_threshold:.9f}")

print(
    f"\nSaved: {OUTPUT_PATH}"
)

# ============================================================
# COMPARE AGAINST CHAMPION
# ============================================================

print("\n")
print("=" * 70)
print("COMPARISON")
print("=" * 70)

print(
    "V1 champion DTI : 0.206570046"
)

print(
    f"Thinning DTI    : {best_score:.9f}"
)

improvement = best_score - 0.206570046

print(
    f"Difference      : {improvement:+.9f}"
)

if best_score > 0.206570046:

    print("\n🔥 Spatial thinning improved V1.")

else:

    print("\nNo improvement over V1 champion.")

print("=" * 70)