import numpy as np
import rasterio
from pathlib import Path

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

PRED_PATH = "outputs/validation_probability_map.npy"
LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

POWER = 0.005
RADIUS = 1

# Fine search around current best K=90,000
K_VALUES = list(range(82_000, 98_001, 500))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V1 SPATIAL THINNING — FINE K SEARCH")
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

print(
    f"Validation GT positives: "
    f"{int(gt.sum()):,}"
)


# ============================================================
# SPATIAL THINNING
# EXACT SAME LOGIC AS SCRIPT 49
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

        # Already too close to another selected point
        if occupied[y0:y1, x0:x1].any():
            continue

        selected.append(idx)

        occupied[y0:y1, x0:x1] = 1

        if len(selected) >= K:
            break

    return np.array(selected, dtype=np.int64)


# ============================================================
# VALIDATION SCORES
# ============================================================

val_prediction = prediction[ys, xs]

print("\nApplying power transformation...")

scores = np.power(
    np.clip(val_prediction, 0, 1),
    POWER
)

print(f"Power: {POWER}")
print(f"Radius: {RADIUS} pixel")


# ============================================================
# SEARCH
# ============================================================

results = []

best_score = -1
best_K = None
best_selected = None
best_threshold = None


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"{'K':>8} "
    f"{'Selected':>10} "
    f"{'TP':>8} "
    f"{'Precision':>12} "
    f"{'Recall':>10} "
    f"{'DTI':>14}"
)

for K in K_VALUES:

    selected_idx = spatial_thin(
        scores,
        val_coords,
        K,
        RADIUS
    )

    test_prediction = np.zeros(
        (H, W),
        dtype=np.float64
    )

    selected_ys = ys[selected_idx]
    selected_xs = xs[selected_idx]

    # Use the transformed score as prediction strength
    test_prediction[
        selected_ys,
        selected_xs
    ] = scores[selected_idx]

    dti = fast_metric(
        gt,
        test_prediction
    )

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
        else 0
    )

    recall = (
        tp / int(gt.sum())
        if gt.sum() > 0
        else 0
    )

    threshold = (
        scores[selected_idx[-1]]
        if selected_count > 0
        else 0
    )

    results.append(
        (
            K,
            selected_count,
            tp,
            precision,
            recall,
            dti,
            threshold
        )
    )

    print(
        f"{K:8,d} "
        f"{selected_count:10,d} "
        f"{tp:8,d} "
        f"{precision:12.6f} "
        f"{recall:10.6f} "
        f"{dti:14.9f}"
    )

    if dti > best_score:

        best_score = dti
        best_K = K
        best_selected = selected_idx.copy()
        best_threshold = threshold


# ============================================================
# BEST RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST RESULT")
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
recall = tp / int(gt.sum())

print(f"Best K        : {best_K:,}")
print(f"Radius        : {RADIUS} pixel")
print(f"Power         : {POWER}")
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

output_map = np.zeros(
    (H, W),
    dtype=np.float64
)

output_map[
    selected_ys,
    selected_xs
] = scores[best_selected]

out_path = (
    "outputs/v1_spatial_fine_k_best.npy"
)

np.save(
    out_path,
    output_map
)

print(f"\nSaved best map:")
print(out_path)


# ============================================================
# SAVE CONFIG
# ============================================================

config_path = (
    "outputs/v1_spatial_fine_k_best_config.txt"
)

with open(config_path, "w") as f:

    f.write(
        f"K={best_K}\n"
        f"RADIUS={RADIUS}\n"
        f"POWER={POWER}\n"
        f"SELECTED={len(best_selected)}\n"
        f"TP={tp}\n"
        f"FP={fp}\n"
        f"PRECISION={precision}\n"
        f"RECALL={recall}\n"
        f"THRESHOLD={best_threshold}\n"
        f"DTI={best_score}\n"
    )

print(f"Saved config:")
print(config_path)