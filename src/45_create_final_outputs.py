import numpy as np
import rasterio
from rasterio.warp import transform
import csv
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
PRED_PATH = "outputs/validation_probability_map.npy"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

OUTPUT_TIF = "outputs/final_fault_probability.tif"
OUTPUT_CSV = "outputs/final_fault_locations.csv"
OUTPUT_PNG = "outputs/final_fault_map.png"

K = 44675
POWER = 0.005


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINAL V1 FAULT MAP GENERATION")
print("=" * 70)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    profile = src.profile.copy()

    transform_raster = src.transform
    source_crs = src.crs

H, W = labels.shape

print("\nRaster:", H, "x", W)
print("CRS:", source_crs)


prediction = np.load(
    PRED_PATH
).astype(np.float64)

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

val_prediction = prediction[
    ys,
    xs
]


# ============================================================
# TOP-K
# ============================================================

print("\nSelecting top predictions...")

order = np.argsort(
    val_prediction
)[::-1]

selected = order[:K]

selected_y = ys[selected]
selected_x = xs[selected]

selected_values = np.clip(
    val_prediction[selected],
    0.0,
    1.0
)

threshold = selected_values[-1]

print("K:", K)
print("Power:", POWER)
print("Threshold:", threshold)


# ============================================================
# POWER TRANSFORMATION
# ============================================================

transformed = np.power(
    selected_values,
    POWER
)


# ============================================================
# FINAL MAP
# ============================================================

final_map = np.zeros(
    (H, W),
    dtype=np.float32
)

final_map[
    selected_y,
    selected_x
] = transformed.astype(
    np.float32
)


# ============================================================
# SAVE GEOTIFF
# ============================================================

profile.update(
    dtype=rasterio.float32,
    count=1,
    compress="lzw",
    nodata=0
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


print(
    "\nSaved GeoTIFF:",
    OUTPUT_TIF
)


# ============================================================
# CONVERT PIXELS → MAP COORDINATES
# ============================================================

print("\nConverting coordinates...")

# Raster coordinates are pixel centers.
map_x, map_y = rasterio.transform.xy(
    transform_raster,
    selected_y,
    selected_x,
    offset="center"
)

map_x = np.asarray(
    map_x,
    dtype=np.float64
)

map_y = np.asarray(
    map_y,
    dtype=np.float64
)


# ============================================================
# MAP CRS → WGS84 LAT/LON
# ============================================================

lon, lat = transform(
    source_crs,
    "EPSG:4326",
    map_x.tolist(),
    map_y.tolist()
)

lon = np.asarray(lon)
lat = np.asarray(lat)


# ============================================================
# SAVE CSV
# ============================================================

print("\nSaving CSV...")

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "rank",
            "row",
            "column",
            "latitude",
            "longitude",
            "prediction"
        ]
    )

    for i in range(K):

        writer.writerow(
            [
                i + 1,
                int(selected_y[i]),
                int(selected_x[i]),
                f"{lat[i]:.8f}",
                f"{lon[i]:.8f}",
                f"{transformed[i]:.8f}"
            ]
        )


print(
    "Saved CSV:",
    OUTPUT_CSV
)


# ============================================================
# VISUALIZATION
# ============================================================

print("\nCreating visualization...")

plt.figure(
    figsize=(12, 9)
)

plt.imshow(
    final_map,
    cmap="hot"
)

plt.colorbar(
    label="Prediction"
)

plt.title(
    "V1 Predicted Geological Fault Locations"
)

plt.xlabel(
    "Pixel Column"
)

plt.ylabel(
    "Pixel Row"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200
)

plt.close()


print(
    "Saved PNG:",
    OUTPUT_PNG
)


# ============================================================
# SHOW SAMPLE COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 PREDICTED LOCATIONS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'Latitude':<14}"
    f"{'Longitude':<14}"
    f"{'Prediction':<14}"
)

for i in range(
    min(20, K)
):

    print(
        f"{i+1:<6}"
        f"{lat[i]:<14.8f}"
        f"{lon[i]:<14.8f}"
        f"{transformed[i]:<14.8f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT COMPLETE")
print("=" * 70)

print("\nModel: V1 U-Net")
print("Top-K:", K)
print("Power:", POWER)
print("Threshold:", threshold)

print(
    "\nFiles:"
)

print(
    "GeoTIFF :",
    OUTPUT_TIF
)

print(
    "CSV     :",
    OUTPUT_CSV
)

print(
    "PNG     :",
    OUTPUT_PNG
)

print("\nDONE")