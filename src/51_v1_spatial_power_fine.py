import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

PRED_PATH = "outputs/validation_probability_map.npy"
LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

K = 91_000
RADIUS = 1

# Fine power search
POWER_VALUES = np.arange(0.001, 0.0201, 0.001)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V1 SPATIAL THINNING — FINE POWER SEARCH")
print("=" * 70)

prediction = np.load(PRED_PATH)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)
    H, W = labels.shape

val_coords = np.load(VAL_COORDS_PATH)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print(f"Prediction shape : {prediction.shape}")
print(f"Label shape      : {labels.shape}")
print(f"Validation pixels: {len(val_coords):,}")


# ============================================================
# CANONICAL GROUND TRUTH
# ============================================================

gt = np.zeros((H, W), dtype=np.float64)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)

GT_POSITIVES = int(gt.sum())

print(f"Validation GT positives: {GT_POSITIVES:,}")


# ============================================================
# SPATIAL THINNING
# SAME LOGIC AS SCRIPTS 48/49/50
# ============================================================

def spatial_thin(scores, coords, K, radius):

    order = np.argsort(scores)[::-1]

    selected = []

    occupied = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    for idx in order:

        y, x = coords[idx]

        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)

        x0 = max(0, x - radius)
        x1 = min(W, x + radius + 1)

        if occupied[y0:y1, x0:x1].any():
            continue

        selected.append(idx)

        occupied[y0:y1, x0:x1] = 1

        if len(selected) >= K:
            break

    return np.array(selected, dtype=np.int64)


# ============================================================
# ORIGINAL VALIDATION SCORES
# ============================================================

val_prediction = prediction[ys, xs]

print("\nStarting power search...")
print(f"K      : {K:,}")
print(f"Radius : {RADIUS} pixel")
print()


# ============================================================
# SEARCH
# ============================================================

best_score = -1.0
best_power = None
best_selected = None
best_threshold = None

results = []

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"{'Power':>8} "
    f"{'Selected':>10} "
    f"{'TP':>8} "
    f"{'Precision':>12} "
    f"{'Recall':>10} "
    f"{'DTI':>14}"
)


for power in POWER_VALUES:

    # --------------------------------------------------------
    # Power transformation
    # --------------------------------------------------------

    scores = np.power(
        np.clip(val_prediction, 0.0, 1.0),
        power
    )

    # --------------------------------------------------------
    # Spatial thinning
    # --------------------------------------------------------

    selected_idx = spatial_thin(
        scores,
        val_coords,
        K,
        RADIUS
    )

    # --------------------------------------------------------
    # Build full prediction raster
    # --------------------------------------------------------

    test_prediction = np.zeros(
        (H, W),
        dtype=np.float64
    )

    selected_ys = ys[selected_idx]
    selected_xs = xs[selected_idx]

    test_prediction[
        selected_ys,
        selected_xs
    ] = scores[selected_idx]

    # --------------------------------------------------------
    # EXACT COMPETITION METRIC
    # --------------------------------------------------------

    dti = fast_metric(
        gt,
        test_prediction
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    tp = int(
        np.sum(
            gt[selected_ys, selected_xs] > 0
        )
    )

    selected_count = len(selected_idx)

    fp = selected_count - tp

    precision = (
        tp / selected_count
        if selected_count > 0
        else 0.0
    )

    recall = (
        tp / GT_POSITIVES
        if GT_POSITIVES > 0
        else 0.0
    )

    threshold = (
        val_prediction[selected_idx[-1]]
        if selected_count > 0
        else 0.0
    )

    results.append(
        (
            float(power),
            selected_count,
            tp,
            precision,
            recall,
            dti,
            threshold
        )
    )

    print(
        f"{power:8.3f} "
        f"{selected_count:10,d} "
        f"{tp:8,d} "
        f"{precision:12.6f} "
        f"{recall:10.6f} "
        f"{dti:14.9f}"
    )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    if dti > best_score:

        best_score = dti
        best_power = float(power)
        best_selected = selected_idx.copy()
        best_threshold = threshold


# ============================================================
# BEST RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST POWER")
print("=" * 70)

selected_ys = ys[best_selected]
selected_xs = xs[best_selected]

tp = int(
    np.sum(
        gt[selected_ys, selected_xs] > 0
    )
)

fp = len(best_selected) - tp

precision = tp / len(best_selected)

recall = tp / GT_POSITIVES

print(f"Best K        : {K:,}")
print(f"Best Radius   : {RADIUS}")
print(f"Best Power    : {best_power:.3f}")
print(f"Selected      : {len(best_selected):,}")
print(f"TP            : {tp:,}")
print(f"FP            : {fp:,}")
print(f"Precision     : {precision:.6f}")
print(f"Recall        : {recall:.6f}")
print(f"Threshold     : {best_threshold:.9f}")
print(f"Exact DTI     : {best_score:.9f}")


# ============================================================
# SAVE BEST MAP
# ============================================================

scores = np.power(
    np.clip(val_prediction, 0.0, 1.0),
    best_power
)

best_map = np.zeros(
    (H, W),
    dtype=np.float64
)

best_map[
    selected_ys,
    selected_xs
] = scores[best_selected]


output_path = (
    "outputs/v1_spatial_power_best.npy"
)

np.save(
    output_path,
    best_map
)

print(f"\nSaved:")
print(output_path)


# ============================================================
# SAVE CONFIG
# ============================================================

config_path = (
    "outputs/v1_spatial_power_best_config.txt"
)

with open(config_path, "w") as f:

    f.write(f"K={K}\n")
    f.write(f"RADIUS={RADIUS}\n")
    f.write(f"POWER={best_power}\n")
    f.write(f"SELECTED={len(best_selected)}\n")
    f.write(f"TP={tp}\n")
    f.write(f"FP={fp}\n")
    f.write(f"PRECISION={precision}\n")
    f.write(f"RECALL={recall}\n")
    f.write(f"THRESHOLD={best_threshold}\n")
    f.write(f"DTI={best_score}\n")

print(config_path)

print("\n" + "=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)