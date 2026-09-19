import os
import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# CONFIG
# ============================================================

PRED_PATH = (
    "outputs/final_test/"
    "reduced_feature_models/"
    "tiled_inference/"
    "V1_19_full_prediction.npy"
)

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

OUTPUT_DIR = (
    "outputs/final_test/"
    "reduced_feature_models/"
    "postprocessing_v2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SEARCH SPACE
# ============================================================

K_VALUES = [
    80000,
    85000,
    90000,
    91000,
    95000,
    100000,
    105000,
    110000,
]

POWER_VALUES = [
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0050,
    0.0070,
    0.0100,
]

RADIUS_VALUES = [
    1,
    2,
]


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("V1_19 POST-PROCESSING OPTIMIZATION")
print("=" * 75)

prediction = np.load(
    PRED_PATH
)

print(
    "Prediction shape:",
    prediction.shape
)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    H, W = labels.shape

val_coords = np.load(
    VAL_COORDS_PATH
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

print(
    f"Validation coordinates: {len(val_coords):,}"
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
).astype(np.float64)

GT_POSITIVES = int(
    gt.sum()
)

print(
    f"Validation GT positives: {GT_POSITIVES:,}"
)


# ============================================================
# VALIDATION SCORES
# ============================================================

val_prediction = prediction[
    ys,
    xs
].astype(np.float64)

val_prediction = np.clip(
    val_prediction,
    0.0,
    1.0
)

print(
    "Validation prediction min:",
    float(val_prediction.min())
)

print(
    "Validation prediction max:",
    float(val_prediction.max())
)

print(
    "Validation prediction mean:",
    float(val_prediction.mean())
)


# ============================================================
# SPATIAL THINNING
# ============================================================

def spatial_thin(
    scores,
    coords,
    K,
    radius
):

    order = np.argsort(
        scores
    )[::-1]

    selected = []

    occupied = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    for idx in order:

        y, x = coords[idx]

        y0 = max(
            0,
            y - radius
        )

        y1 = min(
            H,
            y + radius + 1
        )

        x0 = max(
            0,
            x - radius
        )

        x1 = min(
            W,
            x + radius + 1
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

    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# SEARCH
# ============================================================

best_score = -1.0
best_config = None
best_selected = None
best_scores = None

results = []

total = (
    len(K_VALUES)
    * len(POWER_VALUES)
    * len(RADIUS_VALUES)
)

counter = 0

print()
print(
    f"Total configurations: {total}"
)

print()
print(
    f"{'K':>8} "
    f"{'Power':>8} "
    f"{'Radius':>6} "
    f"{'Selected':>10} "
    f"{'TP':>8} "
    f"{'Precision':>11} "
    f"{'Recall':>10} "
    f"{'DTI':>14}"
)

print("-" * 85)


for radius in RADIUS_VALUES:

    for power in POWER_VALUES:

        # ----------------------------------------------------
        # Power transformation
        # ----------------------------------------------------

        scores = np.power(
            val_prediction,
            power
        )

        for K in K_VALUES:

            counter += 1

            selected_idx = spatial_thin(
                scores,
                val_coords,
                K,
                radius
            )

            selected_ys = ys[
                selected_idx
            ]

            selected_xs = xs[
                selected_idx
            ]

            test_prediction = np.zeros(
                (H, W),
                dtype=np.float64
            )

            test_prediction[
                selected_ys,
                selected_xs
            ] = scores[
                selected_idx
            ]

            dti = fast_metric(
                gt,
                test_prediction
            )

            tp = int(
                np.sum(
                    gt[
                        selected_ys,
                        selected_xs
                    ] > 0
                )
            )

            selected_count = len(
                selected_idx
            )

            fp = (
                selected_count
                - tp
            )

            precision = (
                tp / selected_count
                if selected_count > 0
                else 0.0
            )

            recall = (
                tp / GT_POSITIVES
                if GT_POSITIVES > 0
                else 0.0
            )

            results.append(
                (
                    K,
                    power,
                    radius,
                    selected_count,
                    tp,
                    fp,
                    precision,
                    recall,
                    dti
                )
            )

            print(
                f"{K:8,d} "
                f"{power:8.4f} "
                f"{radius:6d} "
                f"{selected_count:10,d} "
                f"{tp:8,d} "
                f"{precision:11.6f} "
                f"{recall:10.6f} "
                f"{dti:14.9f}"
            )

            if dti > best_score:

                best_score = float(dti)

                best_config = (
                    K,
                    power,
                    radius
                )

                best_selected = (
                    selected_idx.copy()
                )

                best_scores = (
                    scores.copy()
                )


# ============================================================
# BEST
# ============================================================

best_K = best_config[0]
best_power = best_config[1]
best_radius = best_config[2]

selected_ys = ys[
    best_selected
]

selected_xs = xs[
    best_selected
]

best_map = np.zeros(
    (H, W),
    dtype=np.float64
)

best_map[
    selected_ys,
    selected_xs
] = best_scores[
    best_selected
]

tp = int(
    np.sum(
        gt[
            selected_ys,
            selected_xs
        ] > 0
    )
)

fp = (
    len(best_selected)
    - tp
)

precision = (
    tp / len(best_selected)
)

recall = (
    tp / GT_POSITIVES
)


# ============================================================
# SAVE MAP
# ============================================================

map_path = os.path.join(
    OUTPUT_DIR,
    "V1_19_best_postprocessed.npy"
)

np.save(
    map_path,
    best_map
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_array = np.asarray(
    results,
    dtype=np.float64
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "V1_19_postprocessing_search.csv"
)

np.savetxt(
    csv_path,
    results_array,
    delimiter=",",
    header=(
        "K,power,radius,selected,"
        "TP,FP,precision,recall,DTI"
    ),
    comments=""
)


# ============================================================
# SAVE CONFIG
# ============================================================

config_path = os.path.join(
    OUTPUT_DIR,
    "V1_19_best_config.txt"
)

with open(
    config_path,
    "w"
) as f:

    f.write(
        f"K={best_K}\n"
    )

    f.write(
        f"POWER={best_power}\n"
    )

    f.write(
        f"RADIUS={best_radius}\n"
    )

    f.write(
        f"SELECTED={len(best_selected)}\n"
    )

    f.write(
        f"TP={tp}\n"
    )

    f.write(
        f"FP={fp}\n"
    )

    f.write(
        f"PRECISION={precision}\n"
    )

    f.write(
        f"RECALL={recall}\n"
    )

    f.write(
        f"DTI={best_score}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("BEST V1_19 POST-PROCESSING CONFIGURATION")
print("=" * 75)

print(
    f"K          : {best_K:,}"
)

print(
    f"Power      : {best_power:.4f}"
)

print(
    f"Radius     : {best_radius}"
)

print(
    f"Selected   : {len(best_selected):,}"
)

print(
    f"TP         : {tp:,}"
)

print(
    f"FP         : {fp:,}"
)

print(
    f"Precision  : {precision:.6f}"
)

print(
    f"Recall     : {recall:.6f}"
)

print(
    f"Exact DTI  : {best_score:.9f}"
)

print()
print(
    "Previous champion:"
)

print(
    "0.239444373"
)

print()
print(
    "Current fixed-config result:"
)

print(
    "0.243410746"
)

print()
print(
    "Saved best map:"
)

print(
    map_path
)

print()
print(
    "Saved search results:"
)

print(
    csv_path
)

print()
print(
    "Saved configuration:"
)

print(
    config_path
)

print("=" * 75)
