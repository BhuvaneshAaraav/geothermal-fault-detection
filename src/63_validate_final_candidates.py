"""
GeoDAWN — Validation of Final Production Candidates

Purpose
-------
Evaluate the FINAL production candidate-selection procedure
on the spatially held-out validation region.

IMPORTANT
---------
This does NOT retrain or optimize anything.

It evaluates the already-produced:
    outputs/final_test/fault_candidates.csv

against the canonical validation coordinates:
    data/processed/unet/val_coords.npy

using the verified competition metric:
    src/src19_metric.py

Outputs
-------
validation_candidate_summary.txt
validation_candidates.csv
validation_tp.csv
validation_fp.csv
validation_fn.csv
validation_candidate_map.png
validation_score_distribution.png
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CANDIDATE_CSV = (
    "outputs/final_test/fault_candidates.csv"
)

LABEL_PATH = (
    "data/raw/Training_fault_labels.tif"
)

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

# Raw validation prediction from the exact inference pipeline
RAW_PREDICTION_PATH = (
    "outputs/final_test/raw_fault_probability.tif"
)

OUTPUT_DIR = (
    "outputs/final_test/validation"
)

# Verified metric
SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src19_metric import fast_metric


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("GEODAWN FINAL CANDIDATE VALIDATION")
print("=" * 75)


# ============================================================
# LOAD LABELS
# ============================================================

print()
print("=" * 75)
print("LOADING GROUND-TRUTH LABELS")
print("=" * 75)

with rasterio.open(
    LABEL_PATH
) as src:

    labels = src.read(1)

    transform = src.transform
    crs = src.crs

H, W = labels.shape

print(
    "Raster:",
    H,
    "x",
    W
)

print(
    "CRS:",
    crs
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
# CANONICAL VALIDATION GT
# ============================================================

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(
    np.float64
)

gt_positive_count = int(
    gt.sum()
)

print(
    "Validation fault pixels:",
    f"{gt_positive_count:,}"
)


# ============================================================
# LOAD FINAL CANDIDATES
# ============================================================

print()
print("=" * 75)
print("LOADING FINAL PRODUCTION CANDIDATES")
print("=" * 75)

candidates = pd.read_csv(
    CANDIDATE_CSV
)

print(
    "Final candidates:",
    f"{len(candidates):,}"
)


# ============================================================
# LOAD RAW PROBABILITY RASTER
# ============================================================

print()
print("=" * 75)
print("LOADING RAW MODEL PROBABILITY")
print("=" * 75)

with rasterio.open(
    RAW_PREDICTION_PATH
) as src:

    prediction = src.read(1)

print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# FINAL CANDIDATE MAP
# ============================================================

final_candidate_map = np.zeros(
    (H, W),
    dtype=np.float64
)

candidate_rows = (
    candidates["row"]
    .to_numpy(dtype=np.int64)
)

candidate_cols = (
    candidates["column"]
    .to_numpy(dtype=np.int64)
)

candidate_scores = (
    candidates[
        "raw_model_probability"
    ]
    .to_numpy(dtype=np.float64)
)

final_candidate_map[
    candidate_rows,
    candidate_cols
] = candidate_scores


# ============================================================
# RESTRICT FINAL CANDIDATES TO VALIDATION
# ============================================================

print()
print("=" * 75)
print("VALIDATION-ONLY CANDIDATES")
print("=" * 75)

# Fast validation membership map
validation_mask = np.zeros(
    (H, W),
    dtype=np.bool_
)

validation_mask[
    ys,
    xs
] = True


candidate_is_validation = (
    validation_mask[
        candidate_rows,
        candidate_cols
    ]
)

validation_candidates = candidates[
    candidate_is_validation
].copy()

print(
    "Final candidates inside validation region:",
    f"{len(validation_candidates):,}"
)

print(
    "Percentage of final candidates in validation:",
    f"{100 * len(validation_candidates) / len(candidates):.3f}%"
)


# ============================================================
# VALIDATION CANDIDATE MAP
# ============================================================

validation_candidate_map = np.zeros(
    (H, W),
    dtype=np.float64
)

vrows = validation_candidates[
    "row"
].to_numpy(
    dtype=np.int64
)

vcols = validation_candidates[
    "column"
].to_numpy(
    dtype=np.int64
)

vscores = validation_candidates[
    "raw_model_probability"
].to_numpy(
    dtype=np.float64
)

validation_candidate_map[
    vrows,
    vcols
] = vscores


# ============================================================
# EXACT METRIC
# ============================================================

print()
print("=" * 75)
print("EXACT COMPETITION METRIC")
print("=" * 75)

raw_dti = fast_metric(
    gt,
    prediction.astype(np.float64)
)

final_candidate_dti = fast_metric(
    gt,
    validation_candidate_map
)

print(
    "Raw model validation DTI:",
    f"{raw_dti:.9f}"
)

print(
    "Final candidate validation DTI:",
    f"{final_candidate_dti:.9f}"
)


# ============================================================
# BINARY CANDIDATE EVALUATION
# ============================================================

candidate_binary = (
    validation_candidate_map > 0
).astype(
    np.uint8
)

gt_binary = (
    gt > 0
).astype(
    np.uint8
)

tp_map = (
    candidate_binary
    &
    gt_binary
)

fp_map = (
    candidate_binary
    &
    (~gt_binary.astype(bool))
)

fn_map = (
    (~candidate_binary.astype(bool))
    &
    gt_binary.astype(bool)
)

tp = int(
    tp_map.sum()
)

fp = int(
    fp_map.sum()
)

fn = int(
    fn_map.sum()
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    /
    (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


print()
print("=" * 75)
print("BINARY VALIDATION RESULTS")
print("=" * 75)

print(
    "TP:",
    f"{tp:,}"
)

print(
    "FP:",
    f"{fp:,}"
)

print(
    "FN:",
    f"{fn:,}"
)

print(
    "Precision:",
    f"{precision:.6f}"
)

print(
    "Recall:",
    f"{recall:.6f}"
)

print(
    "F1:",
    f"{f1:.6f}"
)


# ============================================================
# DISTANCE TO NEAREST TRUE FAULT
# ============================================================

print()
print("=" * 75)
print("SPATIAL DISTANCE ANALYSIS")
print("=" * 75)

# Validation fault coordinates
fault_y, fault_x = np.where(
    gt_binary > 0
)

candidate_y = vrows
candidate_x = vcols


def nearest_distance_statistics(
    candidate_y,
    candidate_x,
    fault_y,
    fault_x,
    chunk_size=5000
):

    if len(candidate_y) == 0:
        return np.array(
            [],
            dtype=np.float64
        )

    if len(fault_y) == 0:
        return np.full(
            len(candidate_y),
            np.inf,
            dtype=np.float64
        )

    distances = np.empty(
        len(candidate_y),
        dtype=np.float64
    )

    fault_points = np.column_stack(
        [
            fault_y,
            fault_x
        ]
    ).astype(
        np.float64
    )

    for start in range(
        0,
        len(candidate_y),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(candidate_y)
        )

        candidate_points = np.column_stack(
            [
                candidate_y[start:end],
                candidate_x[start:end]
            ]
        ).astype(
            np.float64
        )

        # Squared Euclidean distance
        dy = (
            candidate_points[:, None, 0]
            -
            fault_points[None, :, 0]
        )

        dx = (
            candidate_points[:, None, 1]
            -
            fault_points[None, :, 1]
        )

        dist2 = (
            dy * dy
            +
            dx * dx
        )

        distances[start:end] = np.sqrt(
            np.min(
                dist2,
                axis=1
            )
        )

    return distances


nearest_pixels = nearest_distance_statistics(
    candidate_y,
    candidate_x,
    fault_y,
    fault_x
)

nearest_meters = (
    nearest_pixels * 100.0
)


if len(nearest_meters) > 0:

    print(
        "Median distance:",
        f"{np.median(nearest_meters):.2f} m"
    )

    print(
        "P90 distance:",
        f"{np.percentile(nearest_meters,90):.2f} m"
    )

    print(
        "P95 distance:",
        f"{np.percentile(nearest_meters,95):.2f} m"
    )

    print(
        "Maximum distance:",
        f"{nearest_meters.max():.2f} m"
    )

    for radius_m in [
        100,
        300,
        500,
        1000,
        2000,
        5000
    ]:

        count = int(
            np.sum(
                nearest_meters <= radius_m
            )
        )

        percentage = (
            100.0
            *
            count
            /
            len(nearest_meters)
        )

        print(
            f"Within {radius_m:5d} m:",
            f"{count:7,d}",
            f"({percentage:6.2f}%)"
        )


# ============================================================
# SAVE VALIDATION CANDIDATES
# ============================================================

validation_candidates[
    "is_true_positive"
] = (
    gt_binary[
        vrows,
        vcols
    ] > 0
)

validation_candidates[
    "nearest_fault_distance_m"
] = nearest_meters

validation_candidates.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_candidates.csv"
    ),
    index=False
)


# ============================================================
# SAVE TP / FP / FN
# ============================================================

tp_y, tp_x = np.where(
    tp_map
)

fp_y, fp_x = np.where(
    fp_map
)

fn_y, fn_x = np.where(
    fn_map
)


def save_points(
    path,
    rows,
    cols,
    score_map=None
):

    data = {
        "row": rows,
        "column": cols
    }

    if score_map is not None:

        data[
            "raw_model_probability"
        ] = score_map[
            rows,
            cols
        ]

    out = pd.DataFrame(
        data
    )

    out.to_csv(
        path,
        index=False
    )


save_points(
    os.path.join(
        OUTPUT_DIR,
        "validation_tp.csv"
    ),
    tp_y,
    tp_x,
    validation_candidate_map
)

save_points(
    os.path.join(
        OUTPUT_DIR,
        "validation_fp.csv"
    ),
    fp_y,
    fp_x,
    validation_candidate_map
)

save_points(
    os.path.join(
        OUTPUT_DIR,
        "validation_fn.csv"
    ),
    fn_y,
    fn_x,
    prediction
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("SCORE DISTRIBUTION")
print("=" * 75)

if len(validation_candidates) > 0:

    tp_scores = validation_candidates.loc[
        validation_candidates[
            "is_true_positive"
        ],
        "raw_model_probability"
    ].to_numpy()

    fp_scores = validation_candidates.loc[
        ~validation_candidates[
            "is_true_positive"
        ],
        "raw_model_probability"
    ].to_numpy()

    if len(tp_scores) > 0:

        print()
        print("True-positive candidate scores:")

        print(
            "  Median:",
            f"{np.median(tp_scores):.6f}"
        )

        print(
            "  Mean:",
            f"{tp_scores.mean():.6f}"
        )

        print(
            "  P90:",
            f"{np.percentile(tp_scores,90):.6f}"
        )

        print(
            "  Maximum:",
            f"{tp_scores.max():.6f}"
        )

    if len(fp_scores) > 0:

        print()
        print("False-positive candidate scores:")

        print(
            "  Median:",
            f"{np.median(fp_scores):.6f}"
        )

        print(
            "  Mean:",
            f"{fp_scores.mean():.6f}"
        )

        print(
            "  P90:",
            f"{np.percentile(fp_scores,90):.6f}"
        )

        print(
            "  Maximum:",
            f"{fp_scores.max():.6f}"
        )


# ============================================================
# VALIDATION MAP
# ============================================================

print()
print("=" * 75)
print("GENERATING VALIDATION MAP")
print("=" * 75)

# Only display the validation region.
validation_display = np.full(
    (H, W),
    np.nan,
    dtype=np.float32
)

validation_display[
    ys,
    xs
] = prediction[
    ys,
    xs
]


plt.figure(
    figsize=(12, 9)
)

plt.imshow(
    validation_display,
    origin="upper"
)

plt.title(
    "GeoDAWN Validation Raw Prediction"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.colorbar(
    label="Raw model probability"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_prediction_map.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# TRUE POSITIVE / FALSE POSITIVE MAP
# ============================================================

plt.figure(
    figsize=(12, 9)
)

plt.imshow(
    validation_display,
    origin="upper",
    alpha=0.5
)

if len(tp_y) > 0:

    plt.scatter(
        tp_x,
        tp_y,
        s=3,
        label="TP"
    )

if len(fp_y) > 0:

    plt.scatter(
        fp_x,
        fp_y,
        s=1,
        label="FP"
    )

plt.title(
    "GeoDAWN Validation Candidates"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_candidate_map.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# SCORE HISTOGRAM
# ============================================================

if len(validation_candidates) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    if len(tp_scores) > 0:

        plt.hist(
            tp_scores,
            bins=40,
            alpha=0.7,
            label="True positives"
        )

    if len(fp_scores) > 0:

        plt.hist(
            fp_scores,
            bins=40,
            alpha=0.7,
            label="False positives"
        )

    plt.xlabel(
        "Raw model probability"
    )

    plt.ylabel(
        "Number of candidates"
    )

    plt.title(
        "Validation Candidate Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "validation_score_distribution.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "validation_candidate_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "GeoDAWN Final Production Candidate "
        "Validation\n"
    )

    f.write(
        "==========================================\n\n"
    )

    f.write(
        f"Validation pixels: "
        f"{len(val_coords):,}\n"
    )

    f.write(
        f"Validation fault pixels: "
        f"{gt_positive_count:,}\n"
    )

    f.write(
        f"Total production candidates: "
        f"{len(candidates):,}\n"
    )

    f.write(
        f"Candidates inside validation region: "
        f"{len(validation_candidates):,}\n\n"
    )

    f.write(
        "Exact competition metric\n"
    )

    f.write(
        "------------------------\n"
    )

    f.write(
        f"Raw model DTI: "
        f"{raw_dti:.9f}\n"
    )

    f.write(
        f"Final candidate DTI: "
        f"{final_candidate_dti:.9f}\n\n"
    )

    f.write(
        "Binary candidate results\n"
    )

    f.write(
        "------------------------\n"
    )

    f.write(
        f"TP: {tp:,}\n"
    )

    f.write(
        f"FP: {fp:,}\n"
    )

    f.write(
        f"FN: {fn:,}\n"
    )

    f.write(
        f"Precision: {precision:.6f}\n"
    )

    f.write(
        f"Recall: {recall:.6f}\n"
    )

    f.write(
        f"F1: {f1:.6f}\n\n"
    )

    if len(nearest_meters) > 0:

        f.write(
            "Distance to nearest validation fault\n"
        )

        f.write(
            "------------------------------------\n"
        )

        f.write(
            f"Median: "
            f"{np.median(nearest_meters):.2f} m\n"
        )

        f.write(
            f"P90: "
            f"{np.percentile(nearest_meters,90):.2f} m\n"
        )

        for radius_m in [
            100,
            300,
            500,
            1000,
            2000,
            5000
        ]:

            count = int(
                np.sum(
                    nearest_meters <= radius_m
                )
            )

            percentage = (
                100.0
                *
                count
                /
                len(nearest_meters)
            )

            f.write(
                f"Within {radius_m} m: "
                f"{count:,} "
                f"({percentage:.2f}%)\n"
            )

    f.write(
        "\nInterpretation\n"
    )

    f.write(
        "--------------\n"
    )

    f.write(
        "This analysis evaluates the already-generated "
        "production candidate map on the spatially "
        "held-out validation coordinates. It does not "
        "retrain the model or optimize the production "
        "candidate parameters.\n\n"
    )

    f.write(
        "A candidate classified as a true positive here "
        "means that its center pixel coincides with a "
        "known validation-label fault pixel. The exact "
        "competition metric additionally accounts for "
        "spatial proximity using its distance weighting.\n\n"
    )

    f.write(
        "Model probabilities are scores and should not "
        "be interpreted as calibrated probabilities of "
        "a geological fault.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print()
print(
    "Raw model DTI:",
    f"{raw_dti:.9f}"
)

print(
    "Final candidate DTI:",
    f"{final_candidate_dti:.9f}"
)

print()
print(
    "TP:",
    f"{tp:,}"
)

print(
    "FP:",
    f"{fp:,}"
)

print(
    "FN:",
    f"{fn:,}"
)

print(
    "Precision:",
    f"{precision:.6f}"
)

print(
    "Recall:",
    f"{recall:.6f}"
)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)