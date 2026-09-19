import numpy as np
import rasterio


REFERENCE = "outputs/validation_probability_map.npy"
STANDALONE = "outputs/test_standalone_inference.tif"
VAL_COORDS = "data/processed/unet/val_coords.npy"


print("=" * 70)
print("VALIDATION-ONLY INFERENCE COMPARISON")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

reference = np.load(
    REFERENCE
).astype(np.float32)

with rasterio.open(
    STANDALONE
) as src:

    standalone = src.read(
        1
    ).astype(np.float32)

coords = np.load(
    VAL_COORDS
)

ys = coords[:, 0]
xs = coords[:, 1]


# ============================================================
# VALIDATION PIXELS ONLY
# ============================================================

ref = reference[
    ys,
    xs
]

new = standalone[
    ys,
    xs
]


# ============================================================
# DIFFERENCE
# ============================================================

diff = np.abs(
    ref - new
)


print()
print("Validation pixels:", len(ref))

print()
print("REFERENCE")
print(
    f"Min    : {ref.min():.6f}"
)
print(
    f"Median : {np.median(ref):.6f}"
)
print(
    f"Mean   : {ref.mean():.6f}"
)
print(
    f"P90    : {np.percentile(ref, 90):.6f}"
)
print(
    f"P99    : {np.percentile(ref, 99):.6f}"
)
print(
    f"Max    : {ref.max():.6f}"
)


print()
print("STANDALONE")
print(
    f"Min    : {new.min():.6f}"
)
print(
    f"Median : {np.median(new):.6f}"
)
print(
    f"Mean   : {new.mean():.6f}"
)
print(
    f"P90    : {np.percentile(new, 90):.6f}"
)
print(
    f"P99    : {np.percentile(new, 99):.6f}"
)
print(
    f"Max    : {new.max():.6f}"
)


print()
print("DIFFERENCE")
print(
    f"Mean absolute : {diff.mean():.8f}"
)
print(
    f"Median        : {np.median(diff):.8f}"
)
print(
    f"P95           : {np.percentile(diff, 95):.8f}"
)
print(
    f"P99           : {np.percentile(diff, 99):.8f}"
)
print(
    f"Maximum       : {diff.max():.8f}"
)


# ============================================================
# CORRELATION
# ============================================================

correlation = np.corrcoef(
    ref,
    new
)[0, 1]

print()
print(
    f"Correlation: {correlation:.8f}"
)


# ============================================================
# THRESHOLDS
# ============================================================

print()
print("THRESHOLD AGREEMENT")
print("-" * 70)

for threshold in [
    0.1,
    0.2,
    0.5,
    0.8,
    0.9
]:

    a = ref >= threshold
    b = new >= threshold

    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    print(
        f">= {threshold:.1f}: "
        f"reference={a.sum():,}, "
        f"standalone={b.sum():,}, "
        f"IoU={iou:.6f}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)