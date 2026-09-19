"""
GeoDAWN — Spatial Clustering of Final Fault Candidates

Input:
    outputs/final_test/fault_candidates.csv

Coordinate system:
    EPSG:32611
    x/y are projected coordinates in meters.

Method:
    DBSCAN spatial clustering.

Outputs:
    cluster_assignments.csv
    cluster_summary.csv
    top_clusters.csv
    cluster_map.png
    top_clusters_map.png
    cluster_analysis_summary.txt
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = (
    "outputs/final_test/fault_candidates.csv"
)

OUTPUT_DIR = (
    "outputs/final_test/clusters"
)

# Distance in meters
EPS_METERS = 1000.0

# Minimum number of candidate points
MIN_SAMPLES = 20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("GEODAWN SPATIAL CLUSTER ANALYSIS")
print("=" * 75)

print()
print("Loading:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print()
print(
    "Candidates:",
    f"{len(df):,}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required = [
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
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# EXTRACT COORDINATES
# ============================================================

xy = df[
    [
        "x",
        "y"
    ]
].to_numpy(
    dtype=np.float64
)

scores = df[
    "raw_model_probability"
].to_numpy(
    dtype=np.float64
)


# ============================================================
# BASIC SPATIAL INFORMATION
# ============================================================

print()
print("=" * 75)
print("SPATIAL INPUT")
print("=" * 75)

print(
    "X range:",
    f"{xy[:,0].min():.2f}",
    "to",
    f"{xy[:,0].max():.2f}",
    "meters"
)

print(
    "Y range:",
    f"{xy[:,1].min():.2f}",
    "to",
    f"{xy[:,1].max():.2f}",
    "meters"
)

print(
    "Clustering distance:",
    f"{EPS_METERS:.0f} m"
)

print(
    "Minimum samples:",
    MIN_SAMPLES
)


# ============================================================
# DBSCAN
# ============================================================

print()
print("=" * 75)
print("RUNNING DBSCAN")
print("=" * 75)

clusterer = DBSCAN(
    eps=EPS_METERS,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    n_jobs=-1
)

labels = clusterer.fit_predict(
    xy
)

df["cluster_id"] = labels


# ============================================================
# BASIC CLUSTER STATISTICS
# ============================================================

unique_labels = np.unique(
    labels
)

noise_mask = (
    labels == -1
)

n_noise = int(
    noise_mask.sum()
)

real_labels = unique_labels[
    unique_labels != -1
]

n_clusters = len(
    real_labels
)

print()
print(
    "Clusters found:",
    f"{n_clusters:,}"
)

print(
    "Noise candidates:",
    f"{n_noise:,}"
)

print(
    "Clustered candidates:",
    f"{len(df) - n_noise:,}"
)

print(
    "Clustered percentage:",
    f"{100 * (len(df)-n_noise)/len(df):.2f}%"
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

print()
print("=" * 75)
print("BUILDING CLUSTER SUMMARY")
print("=" * 75)

cluster_records = []

for cluster_id in real_labels:

    mask = (
        labels == cluster_id
    )

    cluster_xy = xy[mask]
    cluster_scores = scores[mask]

    lats = df.loc[
        mask,
        "latitude"
    ].to_numpy()

    lons = df.loc[
        mask,
        "longitude"
    ].to_numpy()

    count = int(
        mask.sum()
    )

    center_x = float(
        cluster_xy[:, 0].mean()
    )

    center_y = float(
        cluster_xy[:, 1].mean()
    )

    center_lat = float(
        lats.mean()
    )

    center_lon = float(
        lons.mean()
    )

    min_x = float(
        cluster_xy[:, 0].min()
    )

    max_x = float(
        cluster_xy[:, 0].max()
    )

    min_y = float(
        cluster_xy[:, 1].min()
    )

    max_y = float(
        cluster_xy[:, 1].max()
    )

    width = (
        max_x - min_x
    )

    height = (
        max_y - min_y
    )

    area_approx = (
        width * height
    )

    record = {

        "cluster_id":
            int(cluster_id),

        "candidate_count":
            count,

        "mean_probability":
            float(cluster_scores.mean()),

        "median_probability":
            float(np.median(cluster_scores)),

        "max_probability":
            float(cluster_scores.max()),

        "p90_probability":
            float(
                np.percentile(
                    cluster_scores,
                    90
                )
            ),

        "p95_probability":
            float(
                np.percentile(
                    cluster_scores,
                    95
                )
            ),

        "center_x_m":
            center_x,

        "center_y_m":
            center_y,

        "center_longitude":
            center_lon,

        "center_latitude":
            center_lat,

        "min_x_m":
            min_x,

        "max_x_m":
            max_x,

        "min_y_m":
            min_y,

        "max_y_m":
            max_y,

        "width_m":
            width,

        "height_m":
            height,

        "approx_bbox_area_km2":
            area_approx / 1_000_000.0
    }

    cluster_records.append(
        record
    )


cluster_summary = pd.DataFrame(
    cluster_records
)


# ============================================================
# SORT BY SIZE
# ============================================================

largest_clusters = (
    cluster_summary
    .sort_values(
        "candidate_count",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

largest_clusters[
    "size_rank"
] = np.arange(
    1,
    len(largest_clusters) + 1
)


# ============================================================
# SORT BY MAX SCORE
# ============================================================

highest_score_clusters = (
    cluster_summary
    .sort_values(
        "max_probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

highest_score_clusters[
    "score_rank"
] = np.arange(
    1,
    len(highest_score_clusters) + 1
)


# ============================================================
# SAVE CLUSTER ASSIGNMENTS
# ============================================================

assignment_path = os.path.join(
    OUTPUT_DIR,
    "cluster_assignments.csv"
)

df.to_csv(
    assignment_path,
    index=False
)

print()
print(
    "Saved:",
    assignment_path
)


# ============================================================
# SAVE CLUSTER SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "cluster_summary.csv"
)

cluster_summary.sort_values(
    "candidate_count",
    ascending=False
).to_csv(
    summary_path,
    index=False
)

print(
    "Saved:",
    summary_path
)


# ============================================================
# TOP 50 LARGEST CLUSTERS
# ============================================================

top50_size = (
    largest_clusters
    .head(50)
)

top50_size_path = os.path.join(
    OUTPUT_DIR,
    "top_50_largest_clusters.csv"
)

top50_size.to_csv(
    top50_size_path,
    index=False
)

print(
    "Saved:",
    top50_size_path
)


# ============================================================
# TOP 50 HIGHEST-SCORE CLUSTERS
# ============================================================

top50_score = (
    highest_score_clusters
    .head(50)
)

top50_score_path = os.path.join(
    OUTPUT_DIR,
    "top_50_highest_score_clusters.csv"
)

top50_score.to_csv(
    top50_score_path,
    index=False
)

print(
    "Saved:",
    top50_score_path
)


# ============================================================
# PRINT TOP CLUSTERS
# ============================================================

print()
print("=" * 75)
print("TOP 20 LARGEST CLUSTERS")
print("=" * 75)

print(
    largest_clusters[
        [
            "size_rank",
            "cluster_id",
            "candidate_count",
            "mean_probability",
            "max_probability",
            "center_longitude",
            "center_latitude",
            "width_m",
            "height_m"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print("=" * 75)
print("TOP 20 HIGHEST-SCORE CLUSTERS")
print("=" * 75)

print(
    highest_score_clusters[
        [
            "score_rank",
            "cluster_id",
            "candidate_count",
            "mean_probability",
            "max_probability",
            "center_longitude",
            "center_latitude",
            "width_m",
            "height_m"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# CLUSTER SIZE DISTRIBUTION
# ============================================================

cluster_sizes = (
    cluster_summary[
        "candidate_count"
    ].to_numpy()
)

print()
print("=" * 75)
print("CLUSTER SIZE STATISTICS")
print("=" * 75)

if len(cluster_sizes) > 0:

    print(
        "Smallest:",
        int(cluster_sizes.min())
    )

    print(
        "Median:",
        int(np.median(cluster_sizes))
    )

    print(
        "Mean:",
        f"{cluster_sizes.mean():.2f}"
    )

    print(
        "P90:",
        f"{np.percentile(cluster_sizes,90):.1f}"
    )

    print(
        "P99:",
        f"{np.percentile(cluster_sizes,99):.1f}"
    )

    print(
        "Largest:",
        int(cluster_sizes.max())
    )


# ============================================================
# MAP: ALL CLUSTERS
# ============================================================

print()
print("=" * 75)
print("GENERATING CLUSTER MAP")
print("=" * 75)

plt.figure(
    figsize=(12, 9)
)

clustered = (
    labels != -1
)

plt.scatter(
    df.loc[
        noise_mask,
        "longitude"
    ],
    df.loc[
        noise_mask,
        "latitude"
    ],
    s=0.4,
    alpha=0.15,
    label="Noise"
)

plt.scatter(
    df.loc[
        clustered,
        "longitude"
    ],
    df.loc[
        clustered,
        "latitude"
    ],
    c=labels[clustered],
    s=1.0,
    cmap="nipy_spectral"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "GeoDAWN Spatial Clusters of Fault Candidates"
)

plt.tight_layout()

cluster_map_path = os.path.join(
    OUTPUT_DIR,
    "cluster_map.png"
)

plt.savefig(
    cluster_map_path,
    dpi=250
)

plt.close()

print(
    "Saved:",
    cluster_map_path
)


# ============================================================
# TOP CLUSTERS MAP
# ============================================================

TOP_N = min(
    20,
    len(largest_clusters)
)

top_cluster_ids = set(
    largest_clusters
    .head(TOP_N)[
        "cluster_id"
    ].astype(int)
)


top_mask = df[
    "cluster_id"
].isin(
    top_cluster_ids
)


plt.figure(
    figsize=(12, 9)
)

plt.scatter(
    df["longitude"],
    df["latitude"],
    s=0.3,
    alpha=0.08
)

plt.scatter(
    df.loc[
        top_mask,
        "longitude"
    ],
    df.loc[
        top_mask,
        "latitude"
    ],
    c=df.loc[
        top_mask,
        "cluster_id"
    ],
    s=1.5,
    cmap="tab20"
)

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    f"GeoDAWN Top {TOP_N} Largest Candidate Clusters"
)

plt.tight_layout()

top_cluster_map_path = os.path.join(
    OUTPUT_DIR,
    "top_clusters_map.png"
)

plt.savefig(
    top_cluster_map_path,
    dpi=250
)

plt.close()

print(
    "Saved:",
    top_cluster_map_path
)


# ============================================================
# TOP SCORE CANDIDATES BY CLUSTER
# ============================================================

top_score_rows = []

for cluster_id in real_labels:

    mask = (
        labels == cluster_id
    )

    cluster_df = df.loc[
        mask
    ]

    top_row = cluster_df.loc[
        cluster_df[
            "raw_model_probability"
        ].idxmax()
    ]

    top_score_rows.append({

        "cluster_id":
            int(cluster_id),

        "candidate_count":
            int(mask.sum()),

        "highest_probability":
            float(
                top_row[
                    "raw_model_probability"
                ]
            ),

        "top_rank":
            int(
                top_row[
                    "rank"
                ]
            ),

        "top_longitude":
            float(
                top_row[
                    "longitude"
                ]
            ),

        "top_latitude":
            float(
                top_row[
                    "latitude"
                ]
            )
    })


cluster_top_candidates = pd.DataFrame(
    top_score_rows
).sort_values(
    "highest_probability",
    ascending=False
)

cluster_top_candidates.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cluster_top_candidates.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY TEXT
# ============================================================

text_path = os.path.join(
    OUTPUT_DIR,
    "cluster_analysis_summary.txt"
)

with open(
    text_path,
    "w"
) as f:

    f.write(
        "GeoDAWN Spatial Cluster Analysis\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Total candidates: {len(df):,}\n"
    )

    f.write(
        f"DBSCAN eps: {EPS_METERS:.0f} m\n"
    )

    f.write(
        f"DBSCAN min_samples: {MIN_SAMPLES}\n\n"
    )

    f.write(
        f"Clusters: {n_clusters:,}\n"
    )

    f.write(
        f"Noise candidates: {n_noise:,}\n"
    )

    f.write(
        f"Clustered candidates: "
        f"{len(df)-n_noise:,}\n"
    )

    f.write(
        f"Clustered percentage: "
        f"{100*(len(df)-n_noise)/len(df):.2f}%\n\n"
    )

    if len(cluster_sizes) > 0:

        f.write(
            "Cluster size statistics\n"
        )

        f.write(
            "-----------------------\n"
        )

        f.write(
            f"Smallest: "
            f"{int(cluster_sizes.min())}\n"
        )

        f.write(
            f"Median: "
            f"{int(np.median(cluster_sizes))}\n"
        )

        f.write(
            f"Mean: "
            f"{cluster_sizes.mean():.2f}\n"
        )

        f.write(
            f"Largest: "
            f"{int(cluster_sizes.max())}\n\n"
        )

    f.write(
        "Interpretation note\n"
    )

    f.write(
        "-------------------\n"
    )

    f.write(
        "Clusters represent spatial groupings of "
        "model-predicted candidate locations. "
        "They should not be interpreted as confirmed "
        "geological faults without independent "
        "geological or geophysical validation.\n"
    )

    f.write(
        "The DBSCAN distance and minimum-sample "
        "parameters are analytical choices and "
        "should be reported explicitly.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("CLUSTER ANALYSIS COMPLETE")
print("=" * 75)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print("Generated:")

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
print("=" * 75)
print("DONE")
print("=" * 75)