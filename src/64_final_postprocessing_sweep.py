"""
GeoDAWN — Final Post-Processing Validation Sweep

Evaluates spatial post-processing on the CANONICAL
held-out validation coordinates.

IMPORTANT:
- No retraining.
- No parameter fitting on the test raster.
- Uses the verified fast_metric().
- Uses the exact V1 validation prediction.
- Spatial thinning is performed over ALL validation
  candidate pixels before stopping at K.

Experiments:
    Radius:
        0, 1, 2, 3

    Power:
        0.001

    K:
        10k -> 100k

Outputs:
    outputs/final_test/postprocessing/
        postprocessing_sweep.csv
        best_config.txt
        dti_vs_k.png
        dti_vs_radius.png
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

PRED_PATH = (
    "outputs/final_test/raw_fault_probability.tif"
)

LABEL_PATH = (
    "data/raw/Training_fault_labels.tif"
)

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

OUTPUT_DIR = (
    "outputs/final_test/postprocessing"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VERIFIED METRIC
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src19_metric import fast_metric


# ============================================================
# PARAMETERS
# ============================================================

POWER_VALUES = [
    0.001
]

K_VALUES = [
    10_000,
    20_000,
    30_000,
    40_000,
    50_000,
    60_000,
    70_000,
    80_000,
    90_000,
    100_000
]

RADIUS_VALUES = [
    0,
    1,
    2,
    3
]


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 75)
print("GEODAWN FINAL POST-PROCESSING SWEEP")
print("=" * 75)

print()
print("Loading labels...")

with rasterio.open(
    LABEL_PATH
) as src:

    labels = src.read(1)

H, W = labels.shape


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

print(
    "Loading validation coordinates..."
)

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
# CANONICAL GROUND TRUTH
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

print(
    "Validation fault pixels:",
    f"{int(gt.sum()):,}"
)


# ============================================================
# LOAD PREDICTION
# ============================================================

print()
print("Loading raw prediction...")

with rasterio.open(
    PRED_PATH
) as src:

    prediction = src.read(1)

prediction = np.asarray(
    prediction,
    dtype=np.float64
)

val_prediction = prediction[
    ys,
    xs
]

print(
    "Prediction statistics:"
)

print(
    "  min   =",
    f"{val_prediction.min():.9f}"
)

print(
    "  median=",
    f"{np.median(val_prediction):.9f}"
)

print(
    "  mean  =",
    f"{val_prediction.mean():.9f}"
)

print(
    "  P90   =",
    f"{np.percentile(val_prediction,90):.9f}"
)

print(
    "  P99   =",
    f"{np.percentile(val_prediction,99):.9f}"
)

print(
    "  max   =",
    f"{val_prediction.max():.9f}"
)


# ============================================================
# RAW DTI
# ============================================================

raw_dti = fast_metric(
    gt,
    prediction
)

print()
print(
    "Raw validation DTI:",
    f"{raw_dti:.9f}"
)


# ============================================================
# SPATIAL THINNING
# ============================================================

def spatial_thin(
    scores,
    coords,
    K,
    radius,
    height,
    width
):

    order = np.argsort(
        scores
    )[::-1]

    selected = []

    occupied = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for idx in order:

        y, x = coords[idx]

        y0 = max(
            0,
            y - radius
        )

        y1 = min(
            height,
            y + radius + 1
        )

        x0 = max(
            0,
            x - radius
        )

        x1 = min(
            width,
            x + radius + 1
        )

        if occupied[
            y0:y1,
            x0:x1
        ].any():

            continue

        selected.append(
            idx
        )

        occupied[
            y0:y1,
            x0:x1
        ] = 1

        if len(selected) >= K:
            break

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# SWEEP
# ============================================================

results = []

total_experiments = (
    len(POWER_VALUES)
    *
    len(RADIUS_VALUES)
    *
    len(K_VALUES)
)

experiment = 0

print()
print("=" * 75)
print(
    "RUNNING",
    total_experiments,
    "EXPERIMENTS"
)
print("=" * 75)


for power in POWER_VALUES:

    # Monotonic transformation.
    # It changes the metric values but not ranking.
    ranking_scores = np.power(
        np.clip(
            val_prediction,
            0,
            1
        ),
        power
    )

    for radius in RADIUS_VALUES:

        for K in K_VALUES:

            experiment += 1

            selected_idx = spatial_thin(
                ranking_scores,
                val_coords,
                K,
                radius,
                H,
                W
            )

            selected_y = ys[
                selected_idx
            ]

            selected_x = xs[
                selected_idx
            ]

            selected_scores = (
                ranking_scores[
                    selected_idx
                ]
            )

            test_prediction = np.zeros(
                (H, W),
                dtype=np.float64
            )

            test_prediction[
                selected_y,
                selected_x
            ] = selected_scores

            dti = fast_metric(
                gt,
                test_prediction
            )

            gt_selected = (
                labels[
                    selected_y,
                    selected_x
                ] > 0
            )

            tp = int(
                gt_selected.sum()
            )

            selected_count = len(
                selected_idx
            )

            fp = (
                selected_count
                -
                tp
            )

            fn = (
                int(gt.sum())
                -
                tp
            )

            precision = (
                tp / selected_count
                if selected_count
                else 0.0
            )

            recall = (
                tp / int(gt.sum())
                if int(gt.sum())
                else 0.0
            )

            # Raw-score threshold corresponding
            # to the last selected candidate.
            if selected_count > 0:

                threshold = float(
                    val_prediction[
                        selected_idx[-1]
                    ]
                )

            else:

                threshold = 0.0

            results.append({

                "power":
                    power,

                "radius":
                    radius,

                "K_requested":
                    K,

                "K_selected":
                    selected_count,

                "threshold_raw_probability":
                    threshold,

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,

                "precision":
                    precision,

                "recall":
                    recall,

                "DTI":
                    dti
            })

            print(
                f"[{experiment:3d}/{total_experiments}] "
                f"radius={radius} "
                f"K={K:6,d} "
                f"DTI={dti:.9f} "
                f"TP={tp:5,d}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "DTI",
    ascending=False
).reset_index(
    drop=True
)

results_path = os.path.join(
    OUTPUT_DIR,
    "postprocessing_sweep.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]

print()
print("=" * 75)
print("BEST CONFIGURATION")
print("=" * 75)

print(
    "Power:",
    best["power"]
)

print(
    "Radius:",
    int(best["radius"])
)

print(
    "K:",
    int(best["K_selected"])
)

print(
    "Threshold:",
    f"{best['threshold_raw_probability']:.9f}"
)

print(
    "TP:",
    int(best["TP"])
)

print(
    "FP:",
    int(best["FP"])
)

print(
    "Precision:",
    f"{best['precision']:.6f}"
)

print(
    "Recall:",
    f"{best['recall']:.6f}"
)

print(
    "DTI:",
    f"{best['DTI']:.9f}"
)


# ============================================================
# TOP 20 CONFIGURATIONS
# ============================================================

print()
print("=" * 75)
print("TOP 20 CONFIGURATIONS")
print("=" * 75)

print(
    results_df[
        [
            "radius",
            "K_selected",
            "threshold_raw_probability",
            "TP",
            "precision",
            "recall",
            "DTI"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE BEST CONFIG
# ============================================================

best_path = os.path.join(
    OUTPUT_DIR,
    "best_config.txt"
)

with open(
    best_path,
    "w"
) as f:

    f.write(
        "GeoDAWN Final Post-Processing Best Configuration\n"
    )

    f.write(
        "=================================================\n\n"
    )

    f.write(
        f"Raw model DTI: {raw_dti:.9f}\n"
    )

    f.write(
        f"Power: {best['power']}\n"
    )

    f.write(
        f"Spatial radius: {int(best['radius'])}\n"
    )

    f.write(
        f"K: {int(best['K_selected'])}\n"
    )

    f.write(
        f"Raw probability threshold: "
        f"{best['threshold_raw_probability']:.9f}\n"
    )

    f.write(
        f"TP: {int(best['TP'])}\n"
    )

    f.write(
        f"FP: {int(best['FP'])}\n"
    )

    f.write(
        f"FN: {int(best['FN'])}\n"
    )

    f.write(
        f"Precision: {best['precision']:.9f}\n"
    )

    f.write(
        f"Recall: {best['recall']:.9f}\n"
    )

    f.write(
        f"Exact DTI: {best['DTI']:.9f}\n"
    )


# ============================================================
# PLOT: DTI VS K
# ============================================================

plt.figure(
    figsize=(10, 7)
)

for radius in RADIUS_VALUES:

    subset = results_df[
        results_df["radius"] == radius
    ].sort_values(
        "K_selected"
    )

    plt.plot(
        subset["K_selected"],
        subset["DTI"],
        marker="o",
        label=f"Radius {radius}"
    )

plt.xlabel(
    "Number of selected candidates (K)"
)

plt.ylabel(
    "Exact Distance-weighted Tversky Index"
)

plt.title(
    "GeoDAWN Post-Processing: DTI vs K"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "dti_vs_k.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# PLOT: DTI VS RADIUS
# ============================================================

best_by_radius = (
    results_df
    .groupby(
        "radius",
        as_index=False
    )["DTI"]
    .max()
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    best_by_radius["radius"],
    best_by_radius["DTI"],
    marker="o"
)

plt.xlabel(
    "Spatial thinning radius (pixels)"
)

plt.ylabel(
    "Best exact DTI"
)

plt.title(
    "Effect of Spatial Thinning Radius"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "dti_vs_radius.png"
    ),
    dpi=220
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("SWEEP COMPLETE")
print("=" * 75)

print()
print(
    "Results:",
    results_path
)

print(
    "Best configuration:",
    best_path
)

print()
print(
    "Raw DTI:",
    f"{raw_dti:.9f}"
)

print(
    "Best post-processed DTI:",
    f"{best['DTI']:.9f}"
)

print()
print("=" * 75)
print("DONE")
print("=" * 75)