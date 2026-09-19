import os

import numpy as np

import rasterio

from src19_metric import fast_metric

# ============================================================

# CONFIG — LOCKED CHAMPION POST-PROCESSING

# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

PRED_DIR = (

    "outputs/final_test/"

    "reduced_feature_models/"

    "tiled_inference"

)

OUTPUT_DIR = (

    "outputs/final_test/"

    "reduced_feature_models/"

    "postprocessing"

)

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

K = 91_000

RADIUS = 1

POWER = 0.001

# ============================================================

# EXPERIMENTS

# ============================================================

EXPERIMENTS = [

    "V1_19",

    "V1_no3",

    "V1_no4",

    "V1_no7",

]

# ============================================================

# LOAD LABELS

# ============================================================

print("=" * 70)

print("RETRAINED MODEL POST-PROCESSING COMPARISON")

print("=" * 70)

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

    H, W = labels.shape

# ============================================================

# LOAD CANONICAL VALIDATION COORDINATES

# ============================================================

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

# EXACT SPATIAL THINNING

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

    return np.array(

        selected,

        dtype=np.int64

    )

# ============================================================

# PROCESS ONE MODEL

# ============================================================

def process_model(name):

    print()

    print("=" * 70)

    print("MODEL:", name)

    print("=" * 70)

    prediction_path = os.path.join(

        PRED_DIR,

        f"{name}_full_prediction.npy"

    )

    print(

        "Loading:",

        prediction_path

    )

    prediction = np.load(

        prediction_path

    )

    print(

        "Prediction shape:",

        prediction.shape

    )

    # --------------------------------------------------------

    # Validation scores

    # --------------------------------------------------------

    val_prediction = prediction[

        ys,

        xs

    ]

    # --------------------------------------------------------

    # Power transformation

    # --------------------------------------------------------

    scores = np.power(

        np.clip(

            val_prediction,

            0.0,

            1.0

        ),

        POWER

    )

    # --------------------------------------------------------

    # Spatial thinning

    # --------------------------------------------------------

    selected_idx = spatial_thin(

        scores,

        val_coords,

        K,

        RADIUS

    )

    # --------------------------------------------------------

    # Build final prediction map

    # --------------------------------------------------------

    test_prediction = np.zeros(

        (H, W),

        dtype=np.float64

    )

    selected_ys = ys[

        selected_idx

    ]

    selected_xs = xs[

        selected_idx

    ]

    test_prediction[

        selected_ys,

        selected_xs

    ] = scores[

        selected_idx

    ]

    # --------------------------------------------------------

    # Exact competition metric

    # --------------------------------------------------------

    dti = fast_metric(

        gt,

        test_prediction

    )

    # --------------------------------------------------------

    # Statistics

    # --------------------------------------------------------

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

    threshold = (

        val_prediction[

            selected_idx[-1]

        ]

        if selected_count > 0

        else 0.0

    )

    # --------------------------------------------------------

    # Save map

    # --------------------------------------------------------

    map_path = os.path.join(

        OUTPUT_DIR,

        f"{name}_postprocessed.npy"

    )

    np.save(

        map_path,

        test_prediction

    )

    # --------------------------------------------------------

    # Save config/results

    # --------------------------------------------------------

    result_path = os.path.join(

        OUTPUT_DIR,

        f"{name}_results.txt"

    )

    with open(

        result_path,

        "w"

    ) as f:

        f.write(

            f"MODEL={name}\n"

        )

        f.write(

            f"K={K}\n"

        )

        f.write(

            f"RADIUS={RADIUS}\n"

        )

        f.write(

            f"POWER={POWER}\n"

        )

        f.write(

            f"SELECTED={selected_count}\n"

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

            f"THRESHOLD={threshold}\n"

        )

        f.write(

            f"DTI={dti}\n"

        )

    print()

    print(

        f"K          : {K:,}"

    )

    print(

        f"Radius     : {RADIUS}"

    )

    print(

        f"Power      : {POWER}"

    )

    print(

        f"Selected   : {selected_count:,}"

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

        f"Threshold  : {threshold:.9f}"

    )

    print(

        f"Exact DTI  : {dti:.9f}"

    )

    print(

        f"Saved map  : {map_path}"

    )

    return dti

# ============================================================

# RUN

# ============================================================

results = {}

for name in EXPERIMENTS:

    results[name] = process_model(

        name

    )

# ============================================================

# SUMMARY

# ============================================================

print()

print("=" * 70)

print("FINAL POST-PROCESSING COMPARISON")

print("=" * 70)

for name, score in results.items():

    print(

        f"{name:10s} "

        f"DTI = {score:.9f}"

    )

print()

print(

    "Current locked champion DTI = 0.239444373"

)

print()

print(

    "Results saved to:"

)

print(

    OUTPUT_DIR

)

print("=" * 70)