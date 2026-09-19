import numpy as np
import rasterio
import csv

from pyproj import Transformer


# ============================================================
# CONFIG
# ============================================================

PRED_PATH = "outputs/validation_probability_map.npy"
LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

OUTPUT_TIF = "outputs/final_fault_probability.tif"
OUTPUT_CSV = "outputs/final_fault_locations.csv"

K = 91_000
RADIUS = 1
POWER = 0.001


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FIXING FINAL PRODUCT SCORES")
print("=" * 70)

prediction = np.load(PRED_PATH)

val_coords = np.load(
    VAL_COORDS_PATH
)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

H, W = labels.shape

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print(f"Raster size      : {H} x {W}")
print(f"Validation pixels: {len(val_coords):,}")
print(f"CRS              : {crs}")

print("\nConfiguration:")
print(f"K       : {K:,}")
print(f"Radius  : {RADIUS}")
print(f"Power   : {POWER}")


# ============================================================
# RAW MODEL PROBABILITIES
# ============================================================

raw_scores = np.clip(
    prediction[ys, xs],
    0.0,
    1.0
)


# ============================================================
# RANKING SCORES
#
# Used ONLY for selecting candidates.
# Do NOT present this as probability.
# ============================================================

ranking_scores = np.power(
    raw_scores,
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


# ============================================================
# SELECT FINAL CANDIDATES
# ============================================================

print("\nSelecting candidates...")

selected_idx = spatial_thin(
    ranking_scores,
    val_coords,
    K,
    RADIUS
)

selected_y = ys[selected_idx]
selected_x = xs[selected_idx]

selected_raw = raw_scores[selected_idx]
selected_ranking = ranking_scores[selected_idx]


# Sort by ranking score
order = np.argsort(
    selected_ranking
)[::-1]

selected_y = selected_y[order]
selected_x = selected_x[order]

selected_raw = selected_raw[order]
selected_ranking = selected_ranking[order]

print(
    f"Selected candidates: "
    f"{len(selected_idx):,}"
)


# ============================================================
# RAW PROBABILITY GEOTIFF
# ============================================================

final_probability_map = np.zeros(
    (H, W),
    dtype=np.float32
)

final_probability_map[
    selected_y,
    selected_x
] = selected_raw.astype(
    np.float32
)

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    height=H,
    width=W,
    compress="deflate",
    nodata=0.0
)

print("\nWriting RAW probability GeoTIFF...")

with rasterio.open(
    OUTPUT_TIF,
    "w",
    **profile
) as dst:

    dst.write(
        final_probability_map,
        1
    )

    dst.set_band_description(
        1,
        "Raw U-Net fault probability"
    )

print(f"Saved: {OUTPUT_TIF}")


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)


# ============================================================
# CSV
# ============================================================

print("\nWriting corrected CSV...")

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
        "raw_model_probability",
        "ranking_score",
        "x",
        "y",
        "longitude",
        "latitude"
    ])

    for rank, (
        y,
        x,
        raw_probability,
        ranking_score
    ) in enumerate(
        zip(
            selected_y,
            selected_x,
            selected_raw,
            selected_ranking
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
            float(raw_probability),
            float(ranking_score),
            float(map_x),
            float(map_y),
            float(lon),
            float(lat)
        ])

print(f"Saved: {OUTPUT_CSV}")


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED FINAL PRODUCT")
print("=" * 70)

print(f"Candidates : {len(selected_idx):,}")

print("\nRaw U-Net probability statistics:")
print(
    f"Min    : {selected_raw.min():.6f}"
)
print(
    f"Median : {np.median(selected_raw):.6f}"
)
print(
    f"Mean   : {selected_raw.mean():.6f}"
)
print(
    f"Max    : {selected_raw.max():.6f}"
)

print("\nTop 10 candidates:")
print("-" * 70)

for i in range(
    min(10, len(selected_y))
):

    y = int(selected_y[i])
    x = int(selected_x[i])

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
        f"Raw probability={selected_raw[i]:.6f}, "
        f"Ranking score={selected_ranking[i]:.6f}"
    )

print("\nIMPORTANT:")
print(
    "Raw model probability is the model output."
)
print(
    "Ranking score is only a post-processing score."
)
print(
    "Neither constitutes geological confirmation."
)

print("\nDONE.")