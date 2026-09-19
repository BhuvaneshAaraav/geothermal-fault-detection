import numpy as np
import rasterio

from src19_metric import fast_metric


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "data/raw/Training_fault_labels.tif"
VAL_COORDS_PATH = "data/processed/unet/val_coords.npy"

V1_PATH = "outputs/validation_probability_map.npy"
V2_PATH = "outputs/hard_negative_validation_probability.npy"
V6_PATH = "outputs/v6_validation_probability.npy"


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 70)
print("CANONICAL MODEL COMPARISON")
print("=" * 70)

print("\nLoading labels...")

with rasterio.open(LABEL_PATH) as src:
    labels = src.read(1)

print("Labels:", labels.shape)


# ============================================================
# LOAD VALIDATION COORDINATES
# ============================================================

val_coords = np.load(
    VAL_COORDS_PATH
)

print(
    "Validation coordinates:",
    val_coords.shape
)


# ============================================================
# BUILD CANONICAL GROUND TRUTH
# ============================================================

H, W = labels.shape

gt = np.zeros(
    (H, W),
    dtype=np.float64
)

mask = np.zeros(
    (H, W),
    dtype=bool
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

mask[ys, xs] = True

gt[ys, xs] = (
    labels[ys, xs] > 0
).astype(np.float64)


print("\nCanonical evaluation region")
print("----------------------------------------")
print("Pixels:", int(mask.sum()))
print("Fault pixels:", int(gt.sum()))


# ============================================================
# MODEL EVALUATOR
# ============================================================

def evaluate_model(name, path):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    prediction = np.load(path).astype(
        np.float64
    )

    print("Loaded:", path)
    print("Shape :", prediction.shape)

    # --------------------------------------------------------
    # Convert prediction into full raster
    # --------------------------------------------------------

    if prediction.shape == (H, W):

        full_prediction = prediction.copy()

    elif prediction.ndim == 1:

        if len(prediction) != len(val_coords):
            raise ValueError(
                f"{name}: prediction length "
                f"{len(prediction)} != "
                f"validation coordinates "
                f"{len(val_coords)}"
            )

        full_prediction = np.zeros(
            (H, W),
            dtype=np.float64
        )

        full_prediction[ys, xs] = prediction

    else:

        raise ValueError(
            f"{name}: unexpected prediction shape "
            f"{prediction.shape}"
        )

    # --------------------------------------------------------
    # Restrict to canonical validation region
    # --------------------------------------------------------

    full_prediction[~mask] = 0.0

    eval_prediction = full_prediction[
        ys,
        xs
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nPrediction statistics")
    print("----------------------------------------")

    print(
        f"Min    : {eval_prediction.min():.9f}"
    )

    print(
        f"Max    : {eval_prediction.max():.9f}"
    )

    print(
        f"Mean   : {eval_prediction.mean():.9f}"
    )

    print(
        f"Median : {np.median(eval_prediction):.9f}"
    )

    print(
        f"P90    : {np.percentile(eval_prediction, 90):.9f}"
    )

    print(
        f"P99    : {np.percentile(eval_prediction, 99):.9f}"
    )

    # --------------------------------------------------------
    # Threshold counts
    # --------------------------------------------------------

    print("\nThreshold counts")
    print("----------------------------------------")

    for threshold in [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]:

        count = np.count_nonzero(
            eval_prediction >= threshold
        )

        print(
            f">= {threshold:.1f}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # EXACT COMPETITION METRIC
    # --------------------------------------------------------

    print("\nCalculating exact DTI...")

    score = fast_metric(
        gt,
        full_prediction
    )

    print(
        f"\n{ name } EXACT DTI = "
        f"{score:.9f}"
    )

    return score


# ============================================================
# EVALUATE MODELS
# ============================================================

results = {}

results["V1"] = evaluate_model(
    "V1",
    V1_PATH
)

results["V2"] = evaluate_model(
    "V2",
    V2_PATH
)

results["V6"] = evaluate_model(
    "V6",
    V6_PATH
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL CANONICAL COMPARISON")
print("=" * 70)

print(
    f"\n{'Model':<10}"
    f"{'Exact DTI':<20}"
)

print("-" * 30)

for name, score in results.items():

    print(
        f"{name:<10}"
        f"{score:<20.9f}"
    )


# ============================================================
# DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("DIFFERENCES")
print("=" * 70)

print(
    f"\nV2 - V1: "
    f"{results['V2'] - results['V1']:+.9f}"
)

print(
    f"V6 - V1: "
    f"{results['V6'] - results['V1']:+.9f}"
)

print(
    f"V6 - V2: "
    f"{results['V6'] - results['V2']:+.9f}"
)


# ============================================================
# BEST RAW MODEL
# ============================================================

best_model = max(
    results,
    key=results.get
)

print("\n" + "=" * 70)
print("RAW DTI COMPARISON COMPLETE")
print("=" * 70)

print(
    f"\nHighest raw DTI among these models: "
    f"{best_model}"
)

print(
    f"DTI: {results[best_model]:.9f}"
)

print("\nIMPORTANT:")
print(
    "All three models were evaluated on the "
    "same validation coordinates using the "
    "verified competition metric."
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)