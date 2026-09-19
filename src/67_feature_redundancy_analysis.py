"""
GEODAWN FEATURE REDUNDANCY ANALYSIS — FIXED

Analyzes redundancy among the 19 geophysical features using:

1. Pearson correlation
2. Spearman correlation
3. Feature statistics
4. Fault-only Pearson correlation
5. Background-only Pearson correlation
6. Highly correlated feature pairs
7. Correlation heatmap

IMPORTANT:
The GeoTIFF contains Float32 NoData values around
-3.4028235e38. These are explicitly removed.
"""

import os

import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================

FEATURE_RASTER = "data/raw/training_features.tif"
LABEL_RASTER = "data/raw/Training_fault_labels.tif"
VAL_COORDS = "data/processed/unet/val_coords.npy"

OUTPUT_DIR = "outputs/final_test/feature_redundancy"

SPEARMAN_MAX_SAMPLES = 300_000

HIGH_CORR = 0.90
VERY_HIGH_CORR = 0.95
NEAR_DUPLICATE_CORR = 0.99

RANDOM_SEED = 42


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg",
    "det_elev_slope",
]


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

print("=" * 75)
print("GEODAWN FEATURE REDUNDANCY ANALYSIS — FIXED")
print("=" * 75)


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

print("\n" + "=" * 75)
print("LOADING CANONICAL VALIDATION COORDINATES")
print("=" * 75)

coords = np.load(VAL_COORDS)

rows = coords[:, 0].astype(np.int64)
cols = coords[:, 1].astype(np.int64)

print("Validation coordinates:", coords.shape)
print("Rows:", rows.min(), "to", rows.max())
print("Cols:", cols.min(), "to", cols.max())


# ============================================================
# LOAD FEATURES
# ============================================================

print("\n" + "=" * 75)
print("LOADING FEATURE RASTER")
print("=" * 75)

with rasterio.open(FEATURE_RASTER) as src:

    print("Raster shape:", (src.count, src.height, src.width))
    print("CRS:", src.crs)

    print("Raster dtype:", src.dtypes[0])
    print("Raster nodata:", src.nodata)

    raster_nodata = src.nodata

    data = src.read()

    X = data[:, rows, cols].T.astype(np.float32)

    del data

print("Validation feature matrix:", X.shape)


# ============================================================
# LOAD LABELS
# ============================================================

print("\n" + "=" * 75)
print("LOADING VALIDATION LABELS")
print("=" * 75)

with rasterio.open(LABEL_RASTER) as src:

    labels = src.read(1)

y = labels[rows, cols]

del labels

print("Validation positive pixels:", int(np.sum(y > 0)))
print("Validation background pixels:", int(np.sum(y == 0)))


# ============================================================
# REMOVE NODATA
# ============================================================

print("\n" + "=" * 75)
print("REMOVING NODATA / INVALID PIXELS")
print("=" * 75)

# Handle the known Float32 NoData sentinel robustly.
#
# Typical value:
# -3.4028235e38
#
# We use both the raster metadata and a safety threshold.

if raster_nodata is not None:

    nodata_mask = np.any(
        X == np.float32(raster_nodata),
        axis=1
    )

else:

    nodata_mask = np.zeros(
        len(X),
        dtype=bool
    )


# Safety protection for Float32 extreme values
extreme_mask = np.any(
    X <= -3.0e38,
    axis=1
)


nonfinite_mask = np.any(
    ~np.isfinite(X),
    axis=1
)


invalid_mask = (
    nodata_mask
    |
    extreme_mask
    |
    nonfinite_mask
)

valid = ~invalid_mask

print("Total pixels:", len(X))
print("NoData pixels:", int(nodata_mask.sum()))
print("Extreme-value pixels:", int(extreme_mask.sum()))
print("Non-finite pixels:", int(nonfinite_mask.sum()))
print("Total invalid:", int(invalid_mask.sum()))
print("Valid pixels:", int(valid.sum()))


# Apply mask

X = X[valid]
y = y[valid]

fault_mask = y > 0
background_mask = y == 0

print("\nAfter cleaning:")
print("Feature matrix:", X.shape)
print("Fault pixels:", int(fault_mask.sum()))
print("Background pixels:", int(background_mask.sum()))


# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("SANITY CHECK")
print("=" * 75)

print("Global minimum:", np.min(X))
print("Global maximum:", np.max(X))
print("Any nonfinite:", np.any(~np.isfinite(X)))
print("Any extreme NoData:", np.any(X <= -3.0e38))

if np.any(~np.isfinite(X)):
    raise RuntimeError(
        "Invalid non-finite values remain."
    )

if np.any(X <= -3.0e38):
    raise RuntimeError(
        "NoData sentinel values remain."
    )

print("✓ Data cleaning successful")


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("FEATURE STATISTICS")
print("=" * 75)

statistics = []

for i, name in enumerate(FEATURE_NAMES):

    values = X[:, i].astype(np.float64)

    statistics.append({
        "band": i + 1,
        "feature": name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    })


stats_df = pd.DataFrame(statistics)

stats_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_statistics.csv"
    ),
    index=False
)

print(stats_df.to_string(index=False))


# ============================================================
# PEARSON
# ============================================================

print("\n" + "=" * 75)
print("PEARSON CORRELATION")
print("=" * 75)

# Use float64 for numerical stability.

X64 = X.astype(np.float64)

pearson = np.corrcoef(X64.T)

pearson_df = pd.DataFrame(
    pearson,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

pearson_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_correlation_pearson.csv"
    )
)

print(
    pearson_df.round(3).to_string()
)


# ============================================================
# FAULT-ONLY PEARSON
# ============================================================

print("\n" + "=" * 75)
print("FAULT-ONLY PEARSON CORRELATION")
print("=" * 75)

X_fault = X64[fault_mask]

fault_pearson = np.corrcoef(
    X_fault.T
)

fault_df = pd.DataFrame(
    fault_pearson,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

fault_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fault_feature_correlation_pearson.csv"
    )
)

print(
    fault_df.round(3).to_string()
)


# ============================================================
# BACKGROUND-ONLY PEARSON
# ============================================================

print("\n" + "=" * 75)
print("BACKGROUND-ONLY PEARSON CORRELATION")
print("=" * 75)

X_background = X64[background_mask]

background_pearson = np.corrcoef(
    X_background.T
)

background_df = pd.DataFrame(
    background_pearson,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

background_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "background_feature_correlation_pearson.csv"
    )
)

print(
    background_df.round(3).to_string()
)


# ============================================================
# SPEARMAN
# ============================================================

print("\n" + "=" * 75)
print("SPEARMAN CORRELATION")
print("=" * 75)

n_samples = len(X)

if n_samples > SPEARMAN_MAX_SAMPLES:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = rng.choice(
        n_samples,
        size=SPEARMAN_MAX_SAMPLES,
        replace=False
    )

    X_spearman = X64[indices]

else:

    X_spearman = X64


print(
    "Samples used:",
    len(X_spearman)
)


n_features = len(FEATURE_NAMES)

spearman_matrix = np.eye(
    n_features,
    dtype=np.float64
)


for i in range(n_features):

    for j in range(i + 1, n_features):

        r, _ = spearmanr(
            X_spearman[:, i],
            X_spearman[:, j]
        )

        spearman_matrix[i, j] = r
        spearman_matrix[j, i] = r


spearman_df = pd.DataFrame(
    spearman_matrix,
    index=FEATURE_NAMES,
    columns=FEATURE_NAMES
)

spearman_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_correlation_spearman.csv"
    )
)

print(
    spearman_df.round(3).to_string()
)


# ============================================================
# FEATURE PAIRS
# ============================================================

print("\n" + "=" * 75)
print("FEATURE CORRELATION PAIRS")
print("=" * 75)

pairs = []

for i in range(n_features):

    for j in range(i + 1, n_features):

        r = pearson[i, j]

        pairs.append({
            "band_1": i + 1,
            "feature_1": FEATURE_NAMES[i],
            "band_2": j + 1,
            "feature_2": FEATURE_NAMES[j],
            "pearson": float(r),
            "abs_pearson": float(abs(r)),
            "spearman": float(
                spearman_matrix[i, j]
            ),
        })


pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    "abs_pearson",
    ascending=False
)


pairs_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_pairs_all.csv"
    ),
    index=False
)


# ============================================================
# HIGH CORRELATION
# ============================================================

high_pairs = pairs_df[
    pairs_df["abs_pearson"] >= HIGH_CORR
]

very_high = pairs_df[
    pairs_df["abs_pearson"] >= VERY_HIGH_CORR
]

near_duplicate = pairs_df[
    pairs_df["abs_pearson"] >= NEAR_DUPLICATE_CORR
]


high_pairs.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_pairs_high_correlation.csv"
    ),
    index=False
)


print(
    "\n|Pearson| >= 0.90:",
    len(high_pairs)
)

print(
    "|Pearson| >= 0.95:",
    len(very_high)
)

print(
    "|Pearson| >= 0.99:",
    len(near_duplicate)
)


print("\nTOP 20 CORRELATED PAIRS:")

print(
    pairs_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# HEATMAP
# ============================================================

print("\n" + "=" * 75)
print("CREATING CORRELATION HEATMAP")
print("=" * 75)

fig, ax = plt.subplots(
    figsize=(15, 13)
)

im = ax.imshow(
    pearson,
    vmin=-1,
    vmax=1
)

ax.set_xticks(
    np.arange(n_features)
)

ax.set_yticks(
    np.arange(n_features)
)

labels = [
    f"{i+1}. {name}"
    for i, name in enumerate(
        FEATURE_NAMES
    )
]

ax.set_xticklabels(
    labels,
    rotation=90
)

ax.set_yticklabels(
    labels
)

ax.set_title(
    "GEODAWN 19-Band Feature Pearson Correlation"
)

fig.colorbar(
    im,
    ax=ax,
    label="Pearson correlation"
)

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "feature_correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    heatmap_path
)


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_redundancy_summary.txt"
)

with open(summary_path, "w") as f:

    f.write(
        "GEODAWN FEATURE REDUNDANCY ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Validation pixels: {len(X)}\n"
    )

    f.write(
        f"Fault pixels: {fault_mask.sum()}\n"
    )

    f.write(
        f"Background pixels: "
        f"{background_mask.sum()}\n\n"
    )

    f.write(
        f"NoData removed: "
        f"{invalid_mask.sum()}\n\n"
    )

    f.write(
        f"|Pearson| >= 0.90: "
        f"{len(high_pairs)}\n"
    )

    f.write(
        f"|Pearson| >= 0.95: "
        f"{len(very_high)}\n"
    )

    f.write(
        f"|Pearson| >= 0.99: "
        f"{len(near_duplicate)}\n\n"
    )

    f.write(
        "Top correlated pairs:\n\n"
    )

    for _, row in pairs_df.head(20).iterrows():

        f.write(
            f"{row['feature_1']} <-> "
            f"{row['feature_2']} : "
            f"Pearson="
            f"{row['pearson']:.6f}, "
            f"Spearman="
            f"{row['spearman']:.6f}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\n✓ NoData values removed")
print("✓ Pearson calculated")
print("✓ Fault-only correlation calculated")
print("✓ Background-only correlation calculated")
print("✓ Spearman calculated")
print("✓ High-correlation pairs identified")
print("✓ Heatmap generated")

print("\nIMPORTANT:")
print(
    "Do NOT remove features based on correlation alone."
)

print(
    "We will combine redundancy + ablation results "
    "before retraining."
)