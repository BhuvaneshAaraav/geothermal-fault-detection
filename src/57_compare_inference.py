import numpy as np
import rasterio


REFERENCE = (
    "outputs/validation_probability_map.npy"
)

STANDALONE = (
    "outputs/test_standalone_inference.tif"
)


print("=" * 70)
print("COMPARING INFERENCE PIPELINES")
print("=" * 70)


# ============================================================
# LOAD REFERENCE
# ============================================================

reference = np.load(
    REFERENCE
).astype(np.float32)


# ============================================================
# LOAD STANDALONE
# ============================================================

with rasterio.open(
    STANDALONE
) as src:

    standalone = src.read(
        1
    ).astype(np.float32)


# ============================================================
# SHAPE
# ============================================================

print(
    "Reference shape :",
    reference.shape
)

print(
    "Standalone shape:",
    standalone.shape
)


if reference.shape != standalone.shape:

    raise ValueError(
        "Prediction shapes do not match."
    )


# ============================================================
# DIFFERENCE
# ============================================================

difference = np.abs(
    reference - standalone
)


print("\nDifference statistics:")
print(
    f"Mean absolute difference : "
    f"{difference.mean():.8f}"
)

print(
    f"Median absolute difference: "
    f"{np.median(difference):.8f}"
)

print(
    f"P95 absolute difference   : "
    f"{np.percentile(difference, 95):.8f}"
)

print(
    f"P99 absolute difference   : "
    f"{np.percentile(difference, 99):.8f}"
)

print(
    f"Maximum difference        : "
    f"{difference.max():.8f}"
)


# ============================================================
# CORRELATION
# ============================================================

r = np.corrcoef(
    reference.ravel(),
    standalone.ravel()
)[0, 1]

print(
    f"\nPearson correlation: {r:.8f}"
)


# ============================================================
# THRESHOLD AGREEMENT
# ============================================================

print("\nThreshold comparison:")

for threshold in [
    0.1,
    0.2,
    0.5,
    0.8,
    0.9
]:

    a = reference >= threshold
    b = standalone >= threshold

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


print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)