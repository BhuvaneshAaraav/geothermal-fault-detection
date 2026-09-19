"""
GeoDAWN — Final Candidate Analysis

Analyzes the final candidate locations produced by:
    src/final_pipeline.py

Outputs:
    1. Candidate statistics
    2. Geographic extent
    3. Probability statistics
    4. Top candidate tables
    5. Spatial density map
    6. Candidate distribution by grid
    7. Analysis text file
"""

import os
import csv

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "outputs/final_test/fault_candidates.csv"
)

PROBABILITY_RASTER = (
    "outputs/final_test/raw_fault_probability.tif"
)

OUTPUT_DIR = (
    "outputs/final_test/analysis"
)

GRID_SIZE = 20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("GEODAWN FINAL CANDIDATE ANALYSIS")
print("=" * 70)

print()
print("Loading:", CSV_PATH)

df = pd.read_csv(
    CSV_PATH
)

print(
    "Candidates:",
    f"{len(df):,}"
)


# ============================================================
# BASIC CHECK
# ============================================================

required_columns = [
    "rank",
    "row",
    "column",
    "raw_model_probability",
    "ranking_score",
    "x",
    "y",
    "longitude",
    "latitude"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# BASIC STATISTICS
# ============================================================

raw = df[
    "raw_model_probability"
].to_numpy()

ranking = df[
    "ranking_score"
].to_numpy()

latitude = df[
    "latitude"
].to_numpy()

longitude = df[
    "longitude"
].to_numpy()


print()
print("=" * 70)
print("PROBABILITY STATISTICS")
print("=" * 70)

print(
    "Minimum:",
    f"{raw.min():.9f}"
)

print(
    "P25:",
    f"{np.percentile(raw, 25):.9f}"
)

print(
    "Median:",
    f"{np.median(raw):.9f}"
)

print(
    "Mean:",
    f"{raw.mean():.9f}"
)

print(
    "P75:",
    f"{np.percentile(raw, 75):.9f}"
)

print(
    "P90:",
    f"{np.percentile(raw, 90):.9f}"
)

print(
    "P95:",
    f"{np.percentile(raw, 95):.9f}"
)

print(
    "P99:",
    f"{np.percentile(raw, 99):.9f}"
)

print(
    "Maximum:",
    f"{raw.max():.9f}"
)


# ============================================================
# THRESHOLD COUNTS
# ============================================================

print()
print("=" * 70)
print("RAW PROBABILITY THRESHOLDS")
print("=" * 70)

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99
]

threshold_rows = []

for threshold in thresholds:

    count = int(
        np.sum(
            raw >= threshold
        )
    )

    percentage = (
        100.0
        *
        count
        /
        len(raw)
    )

    print(
        f">= {threshold:.2f}: "
        f"{count:8,d} "
        f"({percentage:.3f}%)"
    )

    threshold_rows.append(
        [
            threshold,
            count,
            percentage
        ]
    )


threshold_df = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "candidate_count",
        "percentage"
    ]
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "probability_thresholds.csv"
    ),
    index=False
)


# ============================================================
# GEOGRAPHIC EXTENT
# ============================================================

print()
print("=" * 70)
print("GEOGRAPHIC EXTENT")
print("=" * 70)

print(
    "Latitude:",
    f"{latitude.min():.8f}",
    "to",
    f"{latitude.max():.8f}"
)

print(
    "Longitude:",
    f"{longitude.min():.8f}",
    "to",
    f"{longitude.max():.8f}"
)

print(
    "Latitude span:",
    f"{latitude.max() - latitude.min():.8f}"
)

print(
    "Longitude span:",
    f"{longitude.max() - longitude.min():.8f}"
)


# ============================================================
# TOP 100
# ============================================================

print()
print("=" * 70)
print("TOP 20 CANDIDATES")
print("=" * 70)

top20 = df.head(20)

print(
    top20[
        [
            "rank",
            "raw_model_probability",
            "longitude",
            "latitude"
        ]
    ].to_string(
        index=False
    )
)

top20.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_20_candidates.csv"
    ),
    index=False
)

df.head(
    min(100, len(df))
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_100_candidates.csv"
    ),
    index=False
)


# ============================================================
# TOP 500
# ============================================================

df.head(
    min(500, len(df))
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_500_candidates.csv"
    ),
    index=False
)


# ============================================================
# TOP 1000
# ============================================================

df.head(
    min(1000, len(df))
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_1000_candidates.csv"
    ),
    index=False
)


# ============================================================
# LOAD RASTER
# ============================================================

print()
print("=" * 70)
print("LOADING PROBABILITY RASTER")
print("=" * 70)

with rasterio.open(
    PROBABILITY_RASTER
) as src:

    probability = src.read(1)

    transform = src.transform

    crs = src.crs

    height = src.height

    width = src.width

print(
    "Raster:",
    height,
    "x",
    width
)

print(
    "CRS:",
    crs
)


# ============================================================
# CANDIDATE COUNT BY GRID
# ============================================================

print()
print("=" * 70)
print("SPATIAL GRID ANALYSIS")
print("=" * 70)

rows = df[
    "row"
].to_numpy()

cols = df[
    "column"
].to_numpy()

grid_rows = np.linspace(
    0,
    height,
    GRID_SIZE + 1,
    dtype=int
)

grid_cols = np.linspace(
    0,
    width,
    GRID_SIZE + 1,
    dtype=int
)

grid_records = []

for gy in range(
    GRID_SIZE
):

    for gx in range(
        GRID_SIZE
    ):

        y0 = grid_rows[gy]
        y1 = grid_rows[gy + 1]

        x0 = grid_cols[gx]
        x1 = grid_cols[gx + 1]

        mask = (
            (rows >= y0)
            &
            (rows < y1)
            &
            (cols >= x0)
            &
            (cols < x1)
        )

        count = int(
            mask.sum()
        )

        if count > 0:

            mean_probability = float(
                raw[mask].mean()
            )

            max_probability = float(
                raw[mask].max()
            )

        else:

            mean_probability = 0.0
            max_probability = 0.0

        grid_records.append(
            [
                gy,
                gx,
                y0,
                y1,
                x0,
                x1,
                count,
                mean_probability,
                max_probability
            ]
        )

grid_df = pd.DataFrame(
    grid_records,
    columns=[
        "grid_row",
        "grid_column",
        "row_start",
        "row_end",
        "column_start",
        "column_end",
        "candidate_count",
        "mean_raw_probability",
        "max_raw_probability"
    ]
)

grid_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "candidate_grid_analysis.csv"
    ),
    index=False
)


# ============================================================
# DENSITY MAP
# ============================================================

density = np.zeros(
    (GRID_SIZE, GRID_SIZE),
    dtype=np.int32
)

for gy in range(
    GRID_SIZE
):

    for gx in range(
        GRID_SIZE
    ):

        y0 = grid_rows[gy]
        y1 = grid_rows[gy + 1]

        x0 = grid_cols[gx]
        x1 = grid_cols[gx + 1]

        density[
            gy,
            gx
        ] = np.sum(
            (rows >= y0)
            &
            (rows < y1)
            &
            (cols >= x0)
            &
            (cols < x1)
        )


plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    density,
    origin="upper"
)

plt.title(
    "GeoDAWN Candidate Density"
)

plt.xlabel(
    "Grid Column"
)

plt.ylabel(
    "Grid Row"
)

plt.colorbar(
    label="Candidate count"
)

plt.tight_layout()

density_path = os.path.join(
    OUTPUT_DIR,
    "candidate_density_map.png"
)

plt.savefig(
    density_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    density_path
)


# ============================================================
# RAW PROBABILITY HISTOGRAM
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    raw,
    bins=50
)

plt.xlabel(
    "Raw model probability"
)

plt.ylabel(
    "Number of candidates"
)

plt.title(
    "GeoDAWN Candidate Probability Distribution"
)

plt.tight_layout()

hist_path = os.path.join(
    OUTPUT_DIR,
    "candidate_probability_distribution.png"
)

plt.savefig(
    hist_path,
    dpi=200
)

plt.close()

print(
    "Saved:",
    hist_path
)


# ============================================================
# MAP OF ALL CANDIDATES
# ============================================================

plt.figure(
    figsize=(12, 9)
)

plt.scatter(
    longitude,
    latitude,
    s=0.5
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "GeoDAWN Predicted Fault Candidate Locations"
)

plt.tight_layout()

location_map_path = os.path.join(
    OUTPUT_DIR,
    "candidate_locations_map.png"
)

plt.savefig(
    location_map_path,
    dpi=250
)

plt.close()

print(
    "Saved:",
    location_map_path
)


# ============================================================
# TOP 100 MAP
# ============================================================

top = df.head(
    min(
        100,
        len(df)
    )
)

plt.figure(
    figsize=(12, 9)
)

plt.scatter(
    longitude,
    latitude,
    s=0.3,
    alpha=0.15
)

plt.scatter(
    top["longitude"],
    top["latitude"],
    s=8
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "GeoDAWN Top 100 Fault Candidates"
)

plt.tight_layout()

top_map_path = os.path.join(
    OUTPUT_DIR,
    "top_100_candidates_map.png"
)

plt.savefig(
    top_map_path,
    dpi=250
)

plt.close()

print(
    "Saved:",
    top_map_path
)


# ============================================================
# SUMMARY TEXT
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "candidate_analysis_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "GeoDAWN Final Candidate Analysis\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Total candidates: {len(df):,}\n"
    )

    f.write(
        f"Minimum raw probability: "
        f"{raw.min():.9f}\n"
    )

    f.write(
        f"Median raw probability: "
        f"{np.median(raw):.9f}\n"
    )

    f.write(
        f"Mean raw probability: "
        f"{raw.mean():.9f}\n"
    )

    f.write(
        f"P90 raw probability: "
        f"{np.percentile(raw, 90):.9f}\n"
    )

    f.write(
        f"P99 raw probability: "
        f"{np.percentile(raw, 99):.9f}\n"
    )

    f.write(
        f"Maximum raw probability: "
        f"{raw.max():.9f}\n"
    )

    f.write(
        "\nGeographic extent\n"
    )

    f.write(
        "------------------\n"
    )

    f.write(
        f"Latitude: "
        f"{latitude.min():.8f} "
        f"to "
        f"{latitude.max():.8f}\n"
    )

    f.write(
        f"Longitude: "
        f"{longitude.min():.8f} "
        f"to "
        f"{longitude.max():.8f}\n"
    )

    f.write(
        "\nPost-processing\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(
        "K = 91,000\n"
    )

    f.write(
        "Power = 0.001\n"
    )

    f.write(
        "Spatial radius = 1 pixel (~100 m)\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "Interpretation\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "These are model-predicted candidate "
        "fault locations. They are not geological "
        "confirmation of faults.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    "Analysis directory:",
    OUTPUT_DIR
)

print()
print("Generated files:")

for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    print(
        " ",
        filename
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)