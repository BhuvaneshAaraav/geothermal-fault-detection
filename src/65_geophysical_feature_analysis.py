"""
GeoDAWN — 19-Band Geophysical Feature Analysis

Compares feature distributions for:

1. Known validation fault pixels
2. Model true-positive candidate pixels
3. Model false-positive candidate pixels
4. Random validation background pixels

IMPORTANT
---------
This is an observational analysis.

It does NOT prove that a particular geophysical feature
causes a geological fault.

It is intended to identify which input bands show different
distributions between model predictions and known faults.

Outputs
-------
outputs/final_test/geophysical_analysis/

    feature_statistics.csv
    feature_ranking.csv
    feature_distribution.png
    feature_boxplots.png
    correlation_with_fault.csv
    analysis_summary.txt
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = (
    "data/raw/training_features.tif"
)

LABEL_PATH = (
    "data/raw/Training_fault_labels.tif"
)

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

CANDIDATE_CSV = (
    "outputs/final_test/fault_candidates.csv"
)

OUTPUT_DIR = (
    "outputs/final_test/geophysical_analysis"
)

RANDOM_SEED = 42

# Number of background validation pixels
N_BACKGROUND = 20_000

# Number of TP / FP candidates to analyze
MAX_CANDIDATES = 20_000

# Maximum samples used for plots
MAX_PLOT_SAMPLES = 5_000


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("GEODAWN 19-BAND GEOPHYSICAL FEATURE ANALYSIS")
print("=" * 75)


# ============================================================
# LOAD FEATURE RASTER
# ============================================================

print()
print("=" * 75)
print("LOADING 19-BAND FEATURE RASTER")
print("=" * 75)

with rasterio.open(
    FEATURE_PATH
) as src:

    features = src.read()

    descriptions = list(
        src.descriptions
    )

    transform = src.transform
    crs = src.crs

print(
    "Feature shape:",
    features.shape
)

print(
    "CRS:",
    crs
)

n_bands, H, W = features.shape


# ============================================================
# BAND NAMES
# ============================================================

band_names = []

for i in range(
    n_bands
):

    name = descriptions[i]

    if name is None or name == "":

        name = f"band_{i+1}"

    band_names.append(
        name
    )

print()
print("Bands:")

for i, name in enumerate(
    band_names,
    start=1
):

    print(
        f"{i:2d}: {name}"
    )


# ============================================================
# LOAD LABELS
# ============================================================

print()
print("=" * 75)
print("LOADING LABELS")
print("=" * 75)

with rasterio.open(
    LABEL_PATH
) as src:

    labels = src.read(1)

print(
    "Label shape:",
    labels.shape
)


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

print()
print("=" * 75)
print("LOADING VALIDATION COORDINATES")
print("=" * 75)

val_coords = np.load(
    VAL_COORDS_PATH
)

val_coords = np.asarray(
    val_coords,
    dtype=np.int64
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print(
    "Validation pixels:",
    f"{len(val_coords):,}"
)


# ============================================================
# IDENTIFY VALIDATION FAULT / BACKGROUND
# ============================================================

val_labels = (
    labels[
        ys,
        xs
    ] > 0
)

fault_indices = np.where(
    val_labels
)[0]

background_indices = np.where(
    ~val_labels
)[0]

print(
    "Validation fault pixels:",
    f"{len(fault_indices):,}"
)

print(
    "Validation background pixels:",
    f"{len(background_indices):,}"
)


# ============================================================
# SAMPLE BACKGROUND
# ============================================================

background_sample_size = min(
    N_BACKGROUND,
    len(background_indices)
)

background_indices_sample = rng.choice(
    background_indices,
    size=background_sample_size,
    replace=False
)


# ============================================================
# LOAD FINAL CANDIDATES
# ============================================================

print()
print("=" * 75)
print("LOADING FINAL CANDIDATES")
print("=" * 75)

candidates = pd.read_csv(
    CANDIDATE_CSV
)

print(
    "Total candidates:",
    f"{len(candidates):,}"
)


# ============================================================
# RESTRICT CANDIDATES TO VALIDATION
# ============================================================

validation_mask = np.zeros(
    (H, W),
    dtype=np.bool_
)

validation_mask[
    ys,
    xs
] = True

candidate_rows = (
    candidates["row"]
    .to_numpy(
        dtype=np.int64
    )
)

candidate_cols = (
    candidates["column"]
    .to_numpy(
        dtype=np.int64
    )
)

candidate_in_validation = (
    validation_mask[
        candidate_rows,
        candidate_cols
    ]
)

validation_candidates = candidates[
    candidate_in_validation
].copy()

print(
    "Candidates inside validation:",
    f"{len(validation_candidates):,}"
)


# ============================================================
# DETERMINE TP / FP
# ============================================================

vc_rows = validation_candidates[
    "row"
].to_numpy(
    dtype=np.int64
)

vc_cols = validation_candidates[
    "column"
].to_numpy(
    dtype=np.int64
)

vc_is_fault = (
    labels[
        vc_rows,
        vc_cols
    ] > 0
)

validation_candidates[
    "is_true_positive"
] = vc_is_fault


tp_candidates = validation_candidates[
    validation_candidates[
        "is_true_positive"
    ]
].copy()

fp_candidates = validation_candidates[
    ~validation_candidates[
        "is_true_positive"
    ]
].copy()

print(
    "Validation TP candidates:",
    f"{len(tp_candidates):,}"
)

print(
    "Validation FP candidates:",
    f"{len(fp_candidates):,}"
)


# ============================================================
# SAMPLE TP / FP
# ============================================================

def sample_dataframe(
    dataframe,
    maximum
):

    if len(dataframe) <= maximum:

        return dataframe.copy()

    indices = rng.choice(
        len(dataframe),
        size=maximum,
        replace=False
    )

    return dataframe.iloc[
        indices
    ].copy()


tp_sample = sample_dataframe(
    tp_candidates,
    MAX_CANDIDATES
)

fp_sample = sample_dataframe(
    fp_candidates,
    MAX_CANDIDATES
)


# ============================================================
# BUILD COORDINATE GROUPS
# ============================================================

fault_y = ys[
    fault_indices
]

fault_x = xs[
    fault_indices
]

background_y = ys[
    background_indices_sample
]

background_x = xs[
    background_indices_sample
]

tp_y = tp_sample[
    "row"
].to_numpy(
    dtype=np.int64
)

tp_x = tp_sample[
    "column"
].to_numpy(
    dtype=np.int64
)

fp_y = fp_sample[
    "row"
].to_numpy(
    dtype=np.int64
)

fp_x = fp_sample[
    "column"
].to_numpy(
    dtype=np.int64
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print()
print("=" * 75)
print("EXTRACTING FEATURE VALUES")
print("=" * 75)


def extract_features(
    raster,
    rows,
    cols
):

    values = np.empty(
        (
            len(rows),
            raster.shape[0]
        ),
        dtype=np.float64
    )

    for band in range(
        raster.shape[0]
    ):

        band_values = raster[
            band,
            rows,
            cols
        ].astype(
            np.float64
        )

        # Handle raster nodata
        invalid = (
            ~np.isfinite(
                band_values
            )
            |
            (
                np.abs(
                    band_values
                ) > 1e30
            )
        )

        if invalid.any():

            valid_values = band_values[
                ~invalid
            ]

            if len(valid_values) > 0:

                replacement = np.median(
                    valid_values
                )

            else:

                replacement = 0.0

            band_values[
                invalid
            ] = replacement

        values[
            :,
            band
        ] = band_values

    return values


print(
    "Extracting fault pixels..."
)

fault_features = extract_features(
    features,
    fault_y,
    fault_x
)

print(
    "Extracting background pixels..."
)

background_features = extract_features(
    features,
    background_y,
    background_x
)

print(
    "Extracting TP candidates..."
)

tp_features = extract_features(
    features,
    tp_y,
    tp_x
)

print(
    "Extracting FP candidates..."
)

fp_features = extract_features(
    features,
    fp_y,
    fp_x
)


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def robust_statistics(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan
        }

    return {

        "count":
            len(values),

        "mean":
            float(values.mean()),

        "std":
            float(values.std()),

        "median":
            float(np.median(values)),

        "p10":
            float(
                np.percentile(
                    values,
                    10
                )
            ),

        "p25":
            float(
                np.percentile(
                    values,
                    25
                )
            ),

        "p75":
            float(
                np.percentile(
                    values,
                    75
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90
                )
            )
    }


# ============================================================
# FEATURE STATISTICS
# ============================================================

print()
print("=" * 75)
print("CALCULATING FEATURE STATISTICS")
print("=" * 75)

records = []

for band in range(
    n_bands
):

    name = band_names[
        band
    ]

    fault_stats = robust_statistics(
        fault_features[
            :,
            band
        ]
    )

    background_stats = robust_statistics(
        background_features[
            :,
            band
        ]
    )

    tp_stats = robust_statistics(
        tp_features[
            :,
            band
        ]
    )

    fp_stats = robust_statistics(
        fp_features[
            :,
            band
        ]
    )

    # Standardized effect size using background std.
    background_std = (
        background_stats["std"]
    )

    if background_std > 0:

        fault_effect = (
            fault_stats["mean"]
            -
            background_stats["mean"]
        ) / background_std

        tp_effect = (
            tp_stats["mean"]
            -
            background_stats["mean"]
        ) / background_std

        fp_effect = (
            fp_stats["mean"]
            -
            background_stats["mean"]
        ) / background_std

    else:

        fault_effect = np.nan
        tp_effect = np.nan
        fp_effect = np.nan

    # Median-based robust effect
    if background_std > 0:

        fault_median_effect = (
            fault_stats["median"]
            -
            background_stats["median"]
        ) / background_std

        tp_median_effect = (
            tp_stats["median"]
            -
            background_stats["median"]
        ) / background_std

        fp_median_effect = (
            fp_stats["median"]
            -
            background_stats["median"]
        ) / background_std

    else:

        fault_median_effect = np.nan
        tp_median_effect = np.nan
        fp_median_effect = np.nan

    records.append({

        "band":
            band + 1,

        "feature":
            name,

        "fault_mean":
            fault_stats["mean"],

        "background_mean":
            background_stats["mean"],

        "tp_mean":
            tp_stats["mean"],

        "fp_mean":
            fp_stats["mean"],

        "fault_median":
            fault_stats["median"],

        "background_median":
            background_stats["median"],

        "tp_median":
            tp_stats["median"],

        "fp_median":
            fp_stats["median"],

        "fault_effect_size":
            fault_effect,

        "tp_effect_size":
            tp_effect,

        "fp_effect_size":
            fp_effect,

        "fault_median_effect":
            fault_median_effect,

        "tp_median_effect":
            tp_median_effect,

        "fp_median_effect":
            fp_median_effect
    })


stats_df = pd.DataFrame(
    records
)


# ============================================================
# RANK FEATURES
# ============================================================

stats_df[
    "abs_fault_effect"
] = np.abs(
    stats_df[
        "fault_effect_size"
    ]
)

stats_df[
    "abs_tp_effect"
] = np.abs(
    stats_df[
        "tp_effect_size"
    ]
)

stats_df[
    "abs_fp_effect"
] = np.abs(
    stats_df[
        "fp_effect_size"
    ]
)

stats_df = stats_df.sort_values(
    "abs_fault_effect",
    ascending=False
).reset_index(
    drop=True
)

stats_df[
    "fault_effect_rank"
] = np.arange(
    1,
    len(stats_df) + 1
)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "feature_statistics.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)

print(
    "Saved:",
    stats_path
)


# ============================================================
# PRINT FEATURE RANKING
# ============================================================

print()
print("=" * 75)
print("FEATURE RANKING BY ABSOLUTE FAULT/BACKGROUND EFFECT")
print("=" * 75)

print(
    stats_df[
        [
            "fault_effect_rank",
            "feature",
            "fault_mean",
            "background_mean",
            "fault_median",
            "background_median",
            "fault_effect_size"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# TOP FEATURE TABLE
# ============================================================

top_features = stats_df.head(
    min(
        10,
        len(stats_df)
    )
)

top_features[
    [
        "band",
        "feature",
        "fault_effect_size",
        "tp_effect_size",
        "fp_effect_size"
    ]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_ranking.csv"
    ),
    index=False
)


# ============================================================
# CORRELATION WITH FAULT LABEL
# ============================================================

print()
print("=" * 75)
print("FEATURE / FAULT CORRELATION")
print("=" * 75)

# Use sampled background + all fault pixels
corr_fault_values = np.vstack(
    [
        fault_features,
        background_features
    ]
)

corr_labels = np.concatenate(
    [
        np.ones(
            len(fault_features)
        ),
        np.zeros(
            len(background_features)
        )
    ]
)

correlation_records = []

for band in range(
    n_bands
):

    values = corr_fault_values[
        :,
        band
    ]

    valid = np.isfinite(
        values
    )

    if valid.sum() > 1:

        corr = np.corrcoef(
            values[valid],
            corr_labels[valid]
        )[0, 1]

    else:

        corr = np.nan

    correlation_records.append({

        "band":
            band + 1,

        "feature":
            band_names[band],

        "pearson_correlation":
            corr,

        "absolute_correlation":
            abs(corr)
            if np.isfinite(corr)
            else np.nan
    })


correlation_df = pd.DataFrame(
    correlation_records
).sort_values(
    "absolute_correlation",
    ascending=False
)

correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_with_fault.csv"
    ),
    index=False
)

print(
    correlation_df.to_string(
        index=False
    )
)


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

print()
print("=" * 75)
print("GENERATING FEATURE DISTRIBUTION PLOTS")
print("=" * 75)

# Random plotting samples
def sample_array(
    values,
    maximum
):

    if len(values) <= maximum:

        return values

    idx = rng.choice(
        len(values),
        size=maximum,
        replace=False
    )

    return values[
        idx
    ]


plot_fault = sample_array(
    fault_features,
    MAX_PLOT_SAMPLES
)

plot_background = sample_array(
    background_features,
    MAX_PLOT_SAMPLES
)

plot_tp = sample_array(
    tp_features,
    MAX_PLOT_SAMPLES
)

plot_fp = sample_array(
    fp_features,
    MAX_PLOT_SAMPLES
)


# One figure per feature
for band in range(
    n_bands
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        plot_background[:, band],
        bins=40,
        alpha=0.45,
        label="Background"
    )

    plt.hist(
        plot_fault[:, band],
        bins=40,
        alpha=0.45,
        label="Known validation faults"
    )

    if len(plot_tp) > 0:

        plt.hist(
            plot_tp[:, band],
            bins=40,
            alpha=0.45,
            label="Model TP candidates"
        )

    if len(plot_fp) > 0:

        plt.hist(
            plot_fp[:, band],
            bins=40,
            alpha=0.35,
            label="Model FP candidates"
        )

    plt.xlabel(
        band_names[band]
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        f"GeoDAWN Feature Distribution — "
        f"Band {band+1}: {band_names[band]}"
    )

    plt.legend()

    plt.tight_layout()

    filename = (
        f"band_{band+1:02d}_"
        f"{band_names[band]}"
        ".png"
    )

    # Sanitize filename
    filename = filename.replace(
        "/",
        "_"
    ).replace(
        " ",
        "_"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=180
    )

    plt.close()


# ============================================================
# EFFECT SIZE BAR PLOT
# ============================================================

sorted_effect = stats_df.sort_values(
    "fault_effect_size"
)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    sorted_effect["feature"],
    sorted_effect["fault_effect_size"]
)

plt.xlabel(
    "Fault vs background standardized effect"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Geophysical Feature Difference: "
    "Validation Faults vs Background"
)

plt.axvline(
    0,
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "fault_feature_effect_sizes.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# TP VS FP EFFECT
# ============================================================

tp_fp_difference = (
    stats_df[
        "tp_mean"
    ]
    -
    stats_df[
        "fp_mean"
    ]
)

tp_fp_df = pd.DataFrame({

    "feature":
        stats_df["feature"],

    "TP_mean":
        stats_df["tp_mean"],

    "FP_mean":
        stats_df["fp_mean"],

    "TP_minus_FP":
        tp_fp_difference
})

tp_fp_df[
    "absolute_difference"
] = np.abs(
    tp_fp_df[
        "TP_minus_FP"
    ]
)

tp_fp_df = tp_fp_df.sort_values(
    "absolute_difference",
    ascending=False
)

tp_fp_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tp_vs_fp_feature_difference.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "analysis_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "GeoDAWN 19-Band Geophysical Feature Analysis\n"
    )

    f.write(
        "==============================================\n\n"
    )

    f.write(
        f"Number of input bands: {n_bands}\n"
    )

    f.write(
        f"Validation pixels: {len(val_coords):,}\n"
    )

    f.write(
        f"Validation fault pixels: "
        f"{len(fault_indices):,}\n"
    )

    f.write(
        f"Background sample: "
        f"{len(background_features):,}\n"
    )

    f.write(
        f"Validation TP candidate sample: "
        f"{len(tp_features):,}\n"
    )

    f.write(
        f"Validation FP candidate sample: "
        f"{len(fp_features):,}\n\n"
    )

    f.write(
        "Top features by absolute fault/background "
        "effect size\n"
    )

    f.write(
        "------------------------------------------------\n"
    )

    for _, row in stats_df.head(10).iterrows():

        f.write(
            f"{row['feature']}: "
            f"effect={row['fault_effect_size']:.6f}\n"
        )

    f.write(
        "\nInterpretation\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "Effect size describes the difference between "
        "feature distributions in the sampled known "
        "validation faults and sampled validation "
        "background pixels, standardized by the "
        "background standard deviation.\n\n"
    )

    f.write(
        "These statistics describe association, not "
        "causation. A feature with a large effect does "
        "not by itself demonstrate that the feature "
        "causes or uniquely identifies geological "
        "faults.\n\n"
    )

    f.write(
        "TP and FP comparisons describe which input "
        "feature distributions differ between "
        "candidates that coincide with labeled "
        "validation fault pixels and candidates that "
        "do not.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("GEOPHYSICAL FEATURE ANALYSIS COMPLETE")
print("=" * 75)

print()
print(
    "Output directory:"
)

print(
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
print("=" * 75)
print("DONE")
print("=" * 75)