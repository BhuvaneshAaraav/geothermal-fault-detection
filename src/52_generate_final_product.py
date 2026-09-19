import numpy as np
import rasterio
import csv
import matplotlib.pyplot as plt

from pyproj import Transformer


# ============================================================
# FINAL CONFIGURATION — LOCKED
# ============================================================

PRED_PATH = "outputs/validation_probability_map.npy"
LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

OUTPUT_TIF = "outputs/final_fault_probability.tif"
OUTPUT_CSV = "outputs/final_fault_locations.csv"
OUTPUT_PNG = "outputs/final_fault_map.png"

K = 91_000
RADIUS = 1
POWER = 0.001


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINAL GEOTHERMAL FAULT CANDIDATE PRODUCT")
print("=" * 70)

prediction = np.load(PRED_PATH)

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

    H, W = labels.shape

print(f"Raster size      : {H} x {W}")
print(f"Validation pixels: {len(val_coords):,}")
print(f"CRS              : {crs}")

print("\nLOCKED CONFIGURATION")
print(f"K       : {K:,}")
print(f"Radius  : {RADIUS} pixel")
print(f"Power   : {POWER}")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_prediction = np.clip(
    prediction[ys, xs],
    0.0,
    1.0
)

scores = np.power(
    val_prediction,
    POWER
)


# ============================================================
# SPATIAL THINNING
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

    return np.array(
        selected,
        dtype=np.int64
    )


print("\nSelecting final candidate locations...")

selected_idx = spatial_thin(
    scores,
    val_coords,
    K,
    RADIUS
)

selected_y = ys[selected_idx]
selected_x = xs[selected_idx]

selected_scores = scores[selected_idx]

print(
    f"Selected candidates: "
    f"{len(selected_idx):,}"
)


# ============================================================
# SORT FINAL RESULTS
# ============================================================

order = np.argsort(
    selected_scores
)[::-1]

selected_y = selected_y[order]
selected_x = selected_x[order]
selected_scores = selected_scores[order]


# ============================================================
# FINAL PROBABILITY MAP
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
# WRITE GEOTIFF
# ============================================================

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    height=H,
    width=W,
    compress="deflate",
    nodata=0.0
)

print("\nWriting GeoTIFF...")

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
        "Geothermal fault candidate score"
    )

print(f"Saved: {OUTPUT_TIF}")


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

print("\nConverting coordinates...")

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)


# ============================================================
# CSV
# ============================================================

print("Writing CSV...")

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

        # Pixel center
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

print(f"Saved: {OUTPUT_CSV}")


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
    "AI Geothermal Fault Candidate Map"
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

print(f"Saved: {OUTPUT_PNG}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PRODUCT CREATED")
print("=" * 70)

print(f"Candidates : {len(selected_idx):,}")
print(f"K          : {K:,}")
print(f"Radius     : {RADIUS} pixel")
print(f"Power      : {POWER}")
print(f"CRS        : {crs}")

print("\nFiles:")
print(f"1. {OUTPUT_TIF}")
print(f"2. {OUTPUT_CSV}")
print(f"3. {OUTPUT_PNG}")

print("\nTop 10 candidate locations:")
print("-" * 70)

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
        f"{i+1:2d}. "
        f"Lat={lat:.6f}, "
        f"Lon={lon:.6f}, "
        f"Score={score:.6f}"
    )

print("\nDONE.")