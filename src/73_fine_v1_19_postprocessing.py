import os

import numpy as np

import rasterio

from src19_metric import fast_metric

PRED_PATH = (

    "outputs/final_test/"

    "reduced_feature_models/"

    "tiled_inference/"

    "V1_19_full_prediction.npy"

)

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

OUTPUT_DIR = (

    "outputs/final_test/"

    "reduced_feature_models/"

    "postprocessing_v3"

)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fine search around current optimum

K_VALUES = [

    82000,

    83000,

    84000,

    85000,

    86000,

    87000,

    88000,

    89000,

    90000,

    91000,

]

POWER_VALUES = [

    0.00005,

    0.00010,

    0.00015,

    0.00020,

    0.00025,

    0.00030,

    0.00035,

    0.00040,

    0.00045,

    0.00050,

    0.00060,

    0.00070,

    0.00080,

    0.00090,

    0.00100,

]

RADIUS = 1

print("=" * 75)

print("V1_19 FINE POST-PROCESSING SEARCH")

print("=" * 75)

prediction = np.load(PRED_PATH)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

H, W = labels.shape

val_coords = np.load(VAL_COORDS_PATH)

ys = val_coords[:, 0]

xs = val_coords[:, 1]

gt = np.zeros(

    (H, W),

    dtype=np.float64

)

gt[ys, xs] = (

    labels[ys, xs] > 0

).astype(np.float64)

GT_POSITIVES = int(gt.sum())

val_prediction = prediction[

    ys,

    xs

].astype(np.float64)

val_prediction = np.clip(

    val_prediction,

    0.0,

    1.0

)

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

    return np.asarray(

        selected,

        dtype=np.int64

    )

best_dti = -1.0

best = None

results = []

total = len(K_VALUES) * len(POWER_VALUES)

print(f"Configurations: {total}")

print()

print(

    f"{'K':>8} "

    f"{'Power':>10} "

    f"{'TP':>7} "

    f"{'Precision':>11} "

    f"{'Recall':>10} "

    f"{'DTI':>14}"

)

print("-" * 70)

for power in POWER_VALUES:

    scores = np.power(

        val_prediction,

        power

    )

    for K in K_VALUES:

        selected = spatial_thin(

            scores,

            val_coords,

            K,

            RADIUS

        )

        sy = ys[selected]

        sx = xs[selected]

        test_prediction = np.zeros(

            (H, W),

            dtype=np.float64

        )

        test_prediction[sy, sx] = scores[selected]

        dti = fast_metric(

            gt,

            test_prediction

        )

        tp = int(

            np.sum(

                gt[sy, sx] > 0

            )

        )

        count = len(selected)

        fp = count - tp

        precision = (

            tp / count

            if count

            else 0.0

        )

        recall = (

            tp / GT_POSITIVES

        )

        results.append([

            K,

            power,

            count,

            tp,

            fp,

            precision,

            recall,

            dti

        ])

        print(

            f"{K:8,d} "

            f"{power:10.5f} "

            f"{tp:7,d} "

            f"{precision:11.6f} "

            f"{recall:10.6f} "

            f"{dti:14.9f}"

        )

        if dti > best_dti:

            best_dti = float(dti)

            best = {

                "K": K,

                "power": power,

                "radius": RADIUS,

                "selected": selected.copy(),

                "scores": scores.copy(),

                "tp": tp,

                "fp": fp,

                "precision": precision,

                "recall": recall,

            }

# ============================================================

# SAVE BEST MAP

# ============================================================

selected = best["selected"]

scores = best["scores"]

sy = ys[selected]

sx = xs[selected]

best_map = np.zeros(

    (H, W),

    dtype=np.float64

)

best_map[sy, sx] = scores[selected]

np.save(

    os.path.join(

        OUTPUT_DIR,

        "V1_19_best_postprocessed.npy"

    ),

    best_map

)

# ============================================================

# SAVE RESULTS

# ============================================================

results = np.asarray(

    results,

    dtype=np.float64

)

np.savetxt(

    os.path.join(

        OUTPUT_DIR,

        "fine_search.csv"

    ),

    results,

    delimiter=",",

    header=(

        "K,power,selected,TP,FP,"

        "precision,recall,DTI"

    ),

    comments=""

)

with open(

    os.path.join(

        OUTPUT_DIR,

        "best_config.txt"

    ),

    "w"

) as f:

    f.write(

        f"K={best['K']}\n"

    )

    f.write(

        f"POWER={best['power']}\n"

    )

    f.write(

        f"RADIUS={best['radius']}\n"

    )

    f.write(

        f"SELECTED={len(selected)}\n"

    )

    f.write(

        f"TP={best['tp']}\n"

    )

    f.write(

        f"FP={best['fp']}\n"

    )

    f.write(

        f"PRECISION={best['precision']}\n"

    )

    f.write(

        f"RECALL={best['recall']}\n"

    )

    f.write(

        f"DTI={best_dti}\n"

    )

print()

print("=" * 75)

print("BEST FINE-SEARCH RESULT")

print("=" * 75)

print(f"K          : {best['K']:,}")

print(f"Power      : {best['power']:.5f}")

print(f"Radius     : {best['radius']}")

print(f"Selected   : {len(selected):,}")

print(f"TP         : {best['tp']:,}")

print(f"FP         : {best['fp']:,}")

print(f"Precision  : {best['precision']:.6f}")

print(f"Recall     : {best['recall']:.6f}")

print(f"Exact DTI  : {best_dti:.9f}")

print()

print("Previous:")

print("0.243612417")

print("=" * 75)