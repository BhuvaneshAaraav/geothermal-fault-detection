import os
import csv
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer


# ============================================================
# LOCKED V1_19 FINAL CONFIGURATION
# ============================================================

PRED_PATH = (
    "outputs/final_test/reduced_feature_models/"
    "tiled_inference/V1_19_full_prediction.npy"
)

REFERENCE_TIF = "data/raw/Training_fault_labels.tif"

OUTPUT_DIR = (
    "outputs/final_test/reduced_feature_models/"
    "final_product"
)

OUTPUT_TIF = os.path.join(
    OUTPUT_DIR,
    "V1_19_final_fault_candidates.tif"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "V1_19_final_fault_candidates.csv"
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "V1_19_final_fault_map.png"
)

OUTPUT_NPY = os.path.join(
    OUTPUT_DIR,
    "V1_19_final_fault_candidates.npy"
)

K = 84_500
RADIUS = 1
POWER = 0.00005


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 75)
print("V1_19 FINAL GEOTHERMAL FAULT PRODUCT")
print("=" * 75)

print("\nLOCKED CONFIGURATION")
print(f"Prediction : {PRED_PATH}")
print(f"K          : {K:,}")
print(f"Radius     : {RADIUS} pixel")
print(f"Power      : {POWER}")
print("Expected validation DTI : 0.243827623")


# ============================================================
# LOAD PREDICTION
# ============================================================

print("\nLoading V1_19 full-raster prediction...")

prediction = np.load(PRED_PATH)

print("Prediction shape:", prediction.shape)
print("Prediction dtype :", prediction.dtype)


# ============================================================
# LOAD REFERENCE RASTER
# ============================================================

print("\nLoading reference raster...")

with rasterio.open(REFERENCE_TIF) as src:

    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    H = src.height
    W = src.width
    raster_mask = src.read_masks(1) > 0

print(f"Raster size : {H} x {W}")
print(f"CRS        : {crs}")


# ============================================================
# SAFETY CHECKS
# ============================================================

if prediction.shape != (H, W):
    raise ValueError(
        f"Prediction shape {prediction.shape} does not match "
        f"reference raster {(H, W)}"
    )


# ============================================================
# VALID PIXELS
# ============================================================

ys, xs = np.where(raster_mask)

print(f"Valid raster pixels: {len(ys):,}")


# ============================================================
# EXACT V1_19 POST-PROCESSING
#
# Same numerical ordering used by Script 83:
#
# float64
# -> clip [0,1]
# -> power transform
# -> descending ranking
# -> radius-1 spatial thinning
# -> K=84,500
# ============================================================

print("\nApplying locked V1_19 post-processing...")

val_prediction = np.asarray(
    prediction[ys, xs],
    dtype=np.float64
)

val_prediction = np.clip(
    val_prediction,
    0.0,
    1.0
)

scores = np.power(
    val_prediction,
    POWER
)

coords = np.column_stack(
    (ys, xs)
)


# ============================================================
# SPATIAL THINNING
# ============================================================

print("Running spatial thinning...")

order = np.argsort(
    scores,
    kind="stable"
)[::-1]

occupied = np.zeros(
    (H, W),
    dtype=np.uint8
)

selected = []

for idx in order:

    y = int(ys[idx])
    x = int(xs[idx])

    y0 = max(
        0,
        y - RADIUS
    )

    y1 = min(
        H,
        y + RADIUS + 1
    )

    x0 = max(
        0,
        x - RADIUS
    )

    x1 = min(
        W,
        x + RADIUS + 1
    )

    if occupied[
        y0:y1,
        x0:x1
    ].any():

        continue

    selected.append(idx)

    occupied[
        y0:y1,
        x0:x1
    ] = 1

    if len(selected) >= K:
        break


selected_idx = np.asarray(
    selected,
    dtype=np.int64
)

selected_y = ys[selected_idx]
selected_x = xs[selected_idx]
selected_scores = scores[selected_idx]


# ============================================================
# SORT FINAL CANDIDATES
# ============================================================

final_order = np.argsort(
    selected_scores,
    kind="stable"
)[::-1]

selected_y = selected_y[
    final_order
]

selected_x = selected_x[
    final_order
]

selected_scores = selected_scores[
    final_order
]


print(
    f"Selected candidates: "
    f"{len(selected_idx):,}"
)


# ============================================================
# FINAL SPARSE MAP
# ============================================================

final_map = np.zeros(
    (H, W),
    dtype=np.float32
)

final_map[
    selected_y,
    selected_x
] = selected_scores.astype(
    np.float32
)


# ============================================================
# SAVE NPY
# ============================================================

np.save(
    OUTPUT_NPY,
    final_map
)

print("Saved:", OUTPUT_NPY)


# ============================================================
# SAVE GEOTIFF
# ============================================================

print("\nWriting GeoTIFF...")

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    height=H,
    width=W,
    compress="deflate",
    nodata=0.0
)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    **profile
) as dst:

    dst.write(
        final_map,
        1
    )

    dst.set_band_description(
        1,
        "V1_19 geothermal fault candidate score"
    )

print("Saved:", OUTPUT_TIF)


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)


# ============================================================
# SAVE CSV
# ============================================================

print("\nWriting candidate CSV...")

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "rank",
        "row",
        "column",
        "score",
        "x",
        "y",
        "longitude",
        "latitude"
    ])

    for rank, (
        y,
        x,
        score
    ) in enumerate(
        zip(
            selected_y,
            selected_x,
            selected_scores
        ),
        start=1
    ):

        map_x, map_y = rasterio.transform.xy(
            transform,
            int(y),
            int(x),
            offset="center"
        )

        lon, lat = transformer.transform(
            map_x,
            map_y
        )

        writer.writerow([
            rank,
            int(y),
            int(x),
            float(score),
            float(map_x),
            float(map_y),
            float(lon),
            float(lat)
        ])

print("Saved:", OUTPUT_CSV)


# ============================================================
# PNG VISUALIZATION
# ============================================================

print("\nCreating visualization...")

plt.figure(
    figsize=(12, 9)
)

plt.imshow(
    final_map,
    cmap="hot"
)

plt.title(
    "V1_19 AI Geothermal Fault Candidate Map"
)

plt.xlabel(
    "Raster Column"
)

plt.ylabel(
    "Raster Row"
)

plt.colorbar(
    label="Candidate Score"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", OUTPUT_PNG)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("V1_19 FINAL PRODUCT CREATED")
print("=" * 75)

print(f"Candidates : {len(selected_idx):,}")
print(f"K          : {K:,}")
print(f"Radius     : {RADIUS}")
print(f"Power      : {POWER}")
print(f"CRS        : {crs}")

print("\nCandidate score statistics:")
print(f"Min    : {selected_scores.min():.12f}")
print(f"Median : {np.median(selected_scores):.12f}")
print(f"Mean   : {selected_scores.mean():.12f}")
print(f"Max    : {selected_scores.max():.12f}")

print("\nFiles:")
print(f"1. {OUTPUT_TIF}")
print(f"2. {OUTPUT_CSV}")
print(f"3. {OUTPUT_PNG}")
print(f"4. {OUTPUT_NPY}")

print("\nTop 10 candidates:")
print("-" * 75)

for i in range(
    min(10, len(selected_y))
):

    y = int(selected_y[i])
    x = int(selected_x[i])
    score = float(selected_scores[i])

    map_x, map_y = rasterio.transform.xy(
        transform,
        y,
        x,
        offset="center"
    )

    lon, lat = transformer.transform(
        map_x,
        map_y
    )

    print(
        f"{i + 1:4d} | "
        f"row={y:4d} "
        f"col={x:4d} | "
        f"score={score:.8f} | "
        f"lon={lon:.6f} "
        f"lat={lat:.6f}"
    )

print("\nDone.")
