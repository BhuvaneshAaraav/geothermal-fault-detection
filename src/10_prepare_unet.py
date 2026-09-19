import os
import numpy as np
import rasterio


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

OUT_DIR = "data/processed/unet"

# 31 x 31 patch
PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2

# Same spatial split as before
GRID_ROWS = 4
GRID_COLS = 4

VAL_BLOCKS = {1, 7, 9, 10}


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD RASTER METADATA
# ============================================================

print("\n========== LOADING RASTER ==========\n")

with rasterio.open(FEATURES_PATH) as src:
    height = src.height
    width = src.width
    bands = src.count

print("Height:", height)
print("Width :", width)
print("Bands :", bands)


# ============================================================
# CREATE SPATIAL BLOCKS
# ============================================================

print("\n========== CREATING SPATIAL SPLIT ==========\n")

block_h = height // GRID_ROWS
block_w = width // GRID_COLS

block_ids = np.empty((height, width), dtype=np.int8)

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):

        block_id = r * GRID_COLS + c

        y0 = r * block_h
        y1 = (r + 1) * block_h if r < GRID_ROWS - 1 else height

        x0 = c * block_w
        x1 = (c + 1) * block_w if c < GRID_COLS - 1 else width

        block_ids[y0:y1, x0:x1] = block_id


# ============================================================
# READ LABELS
# ============================================================

print("Loading labels...")

with rasterio.open(LABELS_PATH) as src:
    labels = src.read(1)

labels = labels.astype(np.uint8)

print("Total positive pixels:", int((labels == 1).sum()))


# ============================================================
# CREATE TRAIN / VALIDATION MASKS
# ============================================================

train_mask = ~np.isin(block_ids, list(VAL_BLOCKS))
val_mask = np.isin(block_ids, list(VAL_BLOCKS))


# ============================================================
# REMOVE PATCH BOUNDARY LEAKAGE
# ============================================================

print("\n========== APPLYING PATCH SAFETY BUFFER ==========\n")

print("Patch size:", PATCH_SIZE)
print("Patch radius:", RADIUS)
print("Validation blocks:", sorted(VAL_BLOCKS))

# A training center must be at least RADIUS pixels
# away from the validation region.
#
# Likewise, validation centers must be far enough from
# training regions so their complete patches stay inside
# the validation block.

safe_train = train_mask.copy()
safe_val = val_mask.copy()

# Remove outer border
safe_train[:RADIUS, :] = False
safe_train[-RADIUS:, :] = False
safe_train[:, :RADIUS] = False
safe_train[:, -RADIUS:] = False

safe_val[:RADIUS, :] = False
safe_val[-RADIUS:, :] = False
safe_val[:, :RADIUS] = False
safe_val[:, -RADIUS:] = False


# ============================================================
# MORE IMPORTANT:
# REMOVE CENTERS CLOSE TO THE OTHER SPLIT
# ============================================================

# We use scipy distance transform if available.
try:
    from scipy.ndimage import distance_transform_edt

    print("Using scipy distance transform.")

    # Distance from every train pixel to nearest validation pixel
    distance_from_val = distance_transform_edt(train_mask)

    # Distance from every validation pixel to nearest train pixel
    distance_from_train = distance_transform_edt(val_mask)

    safe_train &= distance_from_val > RADIUS
    safe_val &= distance_from_train > RADIUS

except ImportError:
    raise RuntimeError(
        "scipy is required. Install with:\n"
        "pip install scipy"
    )


# ============================================================
# VALID PIXEL MASK
# ============================================================

# Exclude invalid raster values.
#
# We check all 19 bands only indirectly here because
# the actual patch loader will handle individual NoData
# pixels. The center pixel must at least be finite.

print("\n========== FINDING VALID CENTERS ==========\n")

with rasterio.open(FEATURES_PATH) as src:

    # Read in chunks to avoid unnecessary memory usage.
    valid_center = np.ones((height, width), dtype=bool)

    for y0 in range(0, height, 512):

        y1 = min(y0 + 512, height)

        data = src.read(
            window=((y0, y1), (0, width))
        )

        finite = np.all(np.isfinite(data), axis=0)

        valid_center[y0:y1] = finite


safe_train &= valid_center
safe_val &= valid_center


# ============================================================
# TRAIN COORDINATES
# ============================================================

print("\n========== COLLECTING COORDINATES ==========\n")

train_pos = np.argwhere(
    safe_train & (labels == 1)
)

train_neg = np.argwhere(
    safe_train & (labels == 0)
)

val_pos = np.argwhere(
    safe_val & (labels == 1)
)

val_neg = np.argwhere(
    safe_val & (labels == 0)
)

print("Safe train positives :", len(train_pos))
print("Safe train negatives :", len(train_neg))

print("Safe val positives   :", len(val_pos))
print("Safe val negatives   :", len(val_neg))


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print("\n========== BALANCING TRAINING DATA ==========\n")

rng = np.random.default_rng(42)

n_pos = len(train_pos)
n_neg = len(train_neg)

n = min(n_pos, n_neg)

if n == 0:
    raise RuntimeError("No training samples available.")

selected_pos = rng.choice(
    n_pos,
    size=n,
    replace=False
)

selected_neg = rng.choice(
    n_neg,
    size=n,
    replace=False
)

train_coords = np.concatenate(
    [
        train_pos[selected_pos],
        train_neg[selected_neg]
    ],
    axis=0
)

train_labels = np.concatenate(
    [
        np.ones(n, dtype=np.uint8),
        np.zeros(n, dtype=np.uint8)
    ]
)


# Shuffle

shuffle_idx = rng.permutation(len(train_coords))

train_coords = train_coords[shuffle_idx]
train_labels = train_labels[shuffle_idx]


# ============================================================
# VALIDATION DATA
# ============================================================

val_coords = np.concatenate(
    [
        val_pos,
        val_neg
    ],
    axis=0
)

val_labels = np.concatenate(
    [
        np.ones(len(val_pos), dtype=np.uint8),
        np.zeros(len(val_neg), dtype=np.uint8)
    ]
)


# ============================================================
# SAVE
# ============================================================

print("\n========== SAVING ==========\n")

np.save(
    os.path.join(OUT_DIR, "train_coords.npy"),
    train_coords
)

np.save(
    os.path.join(OUT_DIR, "train_labels.npy"),
    train_labels
)

np.save(
    os.path.join(OUT_DIR, "val_coords.npy"),
    val_coords
)

np.save(
    os.path.join(OUT_DIR, "val_labels.npy"),
    val_labels
)


# ============================================================
# SUMMARY
# ============================================================

print("Training coordinates:", train_coords.shape)
print("Training positives  :", int(train_labels.sum()))
print("Training negatives  :", int((train_labels == 0).sum()))

print()

print("Validation coordinates:", val_coords.shape)
print("Validation positives  :", int(val_labels.sum()))
print("Validation negatives  :", int((val_labels == 0).sum()))

print("\nSaved to:", OUT_DIR)

print("\n========== DONE ==========\n")