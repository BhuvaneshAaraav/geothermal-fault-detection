import numpy as np
import rasterio
from scipy.ndimage import label, binary_dilation

# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"
PRED_PATH = "outputs/validation_probability_map.npy"

# ============================================================
# V1 CHAMPION SETTINGS
# ============================================================

K = 44675
POWER = 0.005

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V1 ERROR ANALYSIS")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)
    transform = src.transform
    crs = src.crs

val_coords = np.load(VAL_COORDS_PATH)
prediction = np.load(PRED_PATH).astype(np.float64)

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]

val_prediction = prediction[ys, xs]
val_gt = (labels[ys, xs] > 0)

print("\nRaster:", (H, W))
print("Validation pixels:", len(val_coords))
print("Validation positives:", int(val_gt.sum()))

# ============================================================
# TOP-K SELECTION
# ============================================================

order = np.argsort(val_prediction)[::-1]

selected = order[:K]

selected_y = ys[selected]
selected_x = xs[selected]

selected_values = val_prediction[selected]

threshold = selected_values[-1]

transformed = np.power(
    np.clip(selected_values, 0, 1),
    POWER
)

# ============================================================
# CLASSIFY
# ============================================================

selected_gt = val_gt[selected]

TP = int(np.sum(selected_gt))
FP = int(K - TP)

# All validation GT pixels not selected = FN
FN = int(np.sum(val_gt) - TP)

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\n")
print("=" * 70)
print("V1 CLASSIFICATION")
print("=" * 70)

print(f"Top-K             : {K}")
print(f"Power             : {POWER}")
print(f"Threshold         : {threshold:.9f}")
print(f"TP                : {TP}")
print(f"FP                : {FP}")
print(f"FN                : {FN}")
print(f"Precision         : {precision:.6f}")
print(f"Recall            : {recall:.6f}")

# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]

for p in percentiles:
    print(
        f"P{p:<5}: "
        f"{np.percentile(val_prediction, p):.9f}"
    )

# ============================================================
# TP / FP / FN SCORE DISTRIBUTIONS
# ============================================================

print("\n")
print("=" * 70)
print("TP / FP SCORE DISTRIBUTIONS")
print("=" * 70)

tp_scores = val_prediction[val_gt]

# False positives among the selected Top-K
fp_scores = selected_values[~selected_gt]

# False negatives = GT pixels outside Top-K
selected_mask = np.zeros(len(val_prediction), dtype=bool)
selected_mask[selected] = True

fn_scores = val_prediction[
    val_gt & ~selected_mask
]

def print_stats(name, values):

    print(f"\n{name}")
    print(f"Count : {len(values)}")

    if len(values) == 0:
        return

    print(f"Min   : {values.min():.9f}")
    print(f"Median: {np.median(values):.9f}")
    print(f"Mean  : {values.mean():.9f}")
    print(f"P90   : {np.percentile(values, 90):.9f}")
    print(f"P99   : {np.percentile(values, 99):.9f}")
    print(f"Max   : {values.max():.9f}")

print_stats("ALL GT FAULT PIXELS", tp_scores)
print_stats("SELECTED TRUE POSITIVES", selected_values[selected_gt])
print_stats("SELECTED FALSE POSITIVES", fp_scores)
print_stats("MISSED FAULTS / FALSE NEGATIVES", fn_scores)

# ============================================================
# HOW MANY GT FAULTS EXIST AT DIFFERENT RANK THRESHOLDS?
# ============================================================

print("\n")
print("=" * 70)
print("RECALL AS K CHANGES")
print("=" * 70)

for k in [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    44675,
    50000,
    60000,
    75000,
    100000,
    150000,
    200000,
]:

    k = min(k, len(order))

    idx = order[:k]

    tp_k = int(np.sum(val_gt[idx]))

    recall_k = tp_k / np.sum(val_gt)

    precision_k = tp_k / k

    print(
        f"K={k:<7} "
        f"TP={tp_k:<6} "
        f"Precision={precision_k:.5f} "
        f"Recall={recall_k:.5f}"
    )

# ============================================================
# CREATE 2D MAPS
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING ERROR MAPS")
print("=" * 70)

tp_map = np.zeros((H, W), dtype=np.uint8)
fp_map = np.zeros((H, W), dtype=np.uint8)
fn_map = np.zeros((H, W), dtype=np.uint8)

# TP / FP
for i in range(K):

    y = selected_y[i]
    x = selected_x[i]

    if selected_gt[i]:
        tp_map[y, x] = 1
    else:
        fp_map[y, x] = 1

# FN
fn_indices = np.where(val_gt & ~selected_mask)[0]

fn_map[
    ys[fn_indices],
    xs[fn_indices]
] = 1

np.save("outputs/v1_tp_map.npy", tp_map)
np.save("outputs/v1_fp_map.npy", fp_map)
np.save("outputs/v1_fn_map.npy", fn_map)

print("Saved:")
print("  outputs/v1_tp_map.npy")
print("  outputs/v1_fp_map.npy")
print("  outputs/v1_fn_map.npy")

# ============================================================
# CONNECTED COMPONENT ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SPATIAL ERROR STRUCTURE")
print("=" * 70)

structure = np.ones((3, 3), dtype=np.uint8)

tp_labels, tp_count = label(tp_map, structure=structure)
fp_labels, fp_count = label(fp_map, structure=structure)
fn_labels, fn_count = label(fn_map, structure=structure)

print(f"TP connected components: {tp_count}")
print(f"FP connected components: {fp_count}")
print(f"FN connected components: {fn_count}")

def component_stats(component_map, count):

    sizes = np.bincount(
        component_map.ravel()
    )[1:]

    if len(sizes) == 0:
        return []

    return sorted(
        enumerate(sizes, start=1),
        key=lambda x: x[1],
        reverse=True
    )

print("\nLargest TP components:")
for idx, size in component_stats(tp_labels, tp_count)[:20]:
    print(f"  {size} pixels")

print("\nLargest FP components:")
for idx, size in component_stats(fp_labels, fp_count)[:20]:
    print(f"  {size} pixels")

print("\nLargest FN components:")
for idx, size in component_stats(fn_labels, fn_count)[:20]:
    print(f"  {size} pixels")

# ============================================================
# SPATIAL NEIGHBOR ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SPATIAL NEIGHBOR ANALYSIS")
print("=" * 70)

# Dilate GT slightly.
# This asks whether false positives are at least near real faults.

for radius in [1, 3, 5, 10]:

    size = radius * 2 + 1

    neighborhood = np.ones(
        (size, size),
        dtype=bool
    )

    gt_map = np.zeros(
        (H, W),
        dtype=bool
    )

    gt_map[
        ys[val_gt],
        xs[val_gt]
    ] = True

    near_gt = binary_dilation(
        gt_map,
        structure=neighborhood
    )

    fp_near_gt = np.sum(
        (fp_map > 0) & near_gt
    )

    total_fp = np.sum(fp_map)

    fraction = (
        fp_near_gt / total_fp
        if total_fp > 0
        else 0
    )

    print(
        f"Radius {radius:<3} pixels: "
        f"{fp_near_gt}/{total_fp} "
        f"FP near GT = {fraction:.4%}"
    )

# ============================================================
# GEOREFERENCED ERROR COORDINATES
# ============================================================

print("\n")
print("=" * 70)
print("GEOREFERENCED ERROR LOCATIONS")
print("=" * 70)

from rasterio.warp import transform as rio_transform

# ------------------------------------------------------------
# TP coordinates
# ------------------------------------------------------------

tp_y = selected_y[selected_gt]
tp_x = selected_x[selected_gt]

tp_x_geo, tp_y_geo = rasterio.transform.xy(
    transform,
    tp_y,
    tp_x,
    offset="center"
)

lon_tp, lat_tp = rio_transform(
    crs,
    "EPSG:4326",
    tp_x_geo,
    tp_y_geo
)

# ------------------------------------------------------------
# FP coordinates
# ------------------------------------------------------------

fp_y = selected_y[~selected_gt]
fp_x = selected_x[~selected_gt]

fp_x_geo, fp_y_geo = rasterio.transform.xy(
    transform,
    fp_y,
    fp_x,
    offset="center"
)

lon_fp, lat_fp = rio_transform(
    crs,
    "EPSG:4326",
    fp_x_geo,
    fp_y_geo
)

# ------------------------------------------------------------
# FN coordinates
# ------------------------------------------------------------

fn_y = ys[fn_indices]
fn_x = xs[fn_indices]

fn_x_geo, fn_y_geo = rasterio.transform.xy(
    transform,
    fn_y,
    fn_x,
    offset="center"
)

lon_fn, lat_fn = rio_transform(
    crs,
    "EPSG:4326",
    fn_x_geo,
    fn_y_geo
)

# ============================================================
# SAVE CSV FILES
# ============================================================

import pandas as pd

tp_df = pd.DataFrame({
    "row": tp_y,
    "col": tp_x,
    "latitude": lat_tp,
    "longitude": lon_tp,
    "score": selected_values[selected_gt]
})

fp_df = pd.DataFrame({
    "row": fp_y,
    "col": fp_x,
    "latitude": lat_fp,
    "longitude": lon_fp,
    "score": fp_scores
})

fn_df = pd.DataFrame({
    "row": fn_y,
    "col": fn_x,
    "latitude": lat_fn,
    "longitude": lon_fn,
    "score": fn_scores
})

tp_df.to_csv(
    "outputs/v1_true_positives.csv",
    index=False
)

fp_df.to_csv(
    "outputs/v1_false_positives.csv",
    index=False
)

fn_df.to_csv(
    "outputs/v1_false_negatives.csv",
    index=False
)

print("\nSaved:")
print("  outputs/v1_true_positives.csv")
print("  outputs/v1_false_positives.csv")
print("  outputs/v1_false_negatives.csv")

# ============================================================
# TOP FALSE POSITIVES
# ============================================================

print("\n")
print("=" * 70)
print("HIGHEST-SCORING FALSE POSITIVES")
print("=" * 70)

top_fp_order = np.argsort(fp_scores)[::-1][:30]

for rank, i in enumerate(top_fp_order, start=1):

    print(
        f"{rank:2d}. "
        f"score={fp_scores[i]:.9f} "
        f"lat={lat_fp[i]:.6f} "
        f"lon={lon_fp[i]:.6f}"
    )

# ============================================================
# HIGHEST-SCORING MISSED FAULTS
# ============================================================

print("\n")
print("=" * 70)
print("HIGHEST-SCORING MISSED FAULTS")
print("=" * 70)

top_fn_order = np.argsort(fn_scores)[::-1][:30]

for rank, i in enumerate(top_fn_order, start=1):

    print(
        f"{rank:2d}. "
        f"score={fn_scores[i]:.9f} "
        f"lat={lat_fn[i]:.6f} "
        f"lon={lon_fn[i]:.6f}"
    )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
V1 Champion
-----------
DTI       : 0.206570046
K         : {K}
Power     : {POWER}
TP        : {TP}
FP        : {FP}
FN        : {FN}
Precision : {precision:.6f}
Recall    : {recall:.6f}

The important outputs are:

v1_true_positives.csv
v1_false_positives.csv
v1_false_negatives.csv

and the three spatial maps:

v1_tp_map.npy
v1_fp_map.npy
v1_fn_map.npy
""")

print("=" * 70)