import os

import numpy as np
import rasterio


FEATURE_FILE = "data/raw/training_features.tif"
LABEL_FILE = "data/raw/Training_fault_labels.tif"

OUTPUT_DIR = "data/processed/cnn"

RANDOM_STATE = 42

PATCH_SIZE = 31

# Same spatial split as before
N_BLOCK_ROWS = 4
N_BLOCK_COLS = 4

TRAIN_BLOCKS = {
    0, 2, 3,
    4, 5, 6,
    8, 11,
    12, 13, 14, 15
}

VALIDATION_BLOCKS = {
    1, 7, 9, 10
}


def create_block_map(height, width):

    block_map = np.full(
        (height, width),
        -1,
        dtype=np.int8
    )

    block_height = height // N_BLOCK_ROWS
    block_width = width // N_BLOCK_COLS

    block_id = 0

    for row in range(N_BLOCK_ROWS):

        for col in range(N_BLOCK_COLS):

            row_start = row * block_height
            col_start = col * block_width

            row_end = (
                height
                if row == N_BLOCK_ROWS - 1
                else (row + 1) * block_height
            )

            col_end = (
                width
                if col == N_BLOCK_COLS - 1
                else (col + 1) * block_width
            )

            if block_id in TRAIN_BLOCKS:
                block_map[
                    row_start:row_end,
                    col_start:col_end
                ] = 0

            elif block_id in VALIDATION_BLOCKS:
                block_map[
                    row_start:row_end,
                    col_start:col_end
                ] = 1

            block_id += 1

    return block_map


def get_coordinates(
    labels,
    valid_mask,
    block_map,
    block_value
):

    half = PATCH_SIZE // 2

    region = (
        block_map == block_value
    )

    # Don't allow patches to extend outside the raster.
    safe_region = np.zeros_like(region)

    safe_region[
        half:-half,
        half:-half
    ] = True

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    positive_coords = np.argwhere(
        positive &
        region &
        valid_mask &
        safe_region
    )

    negative_coords = np.argwhere(
        negative &
        region &
        valid_mask &
        safe_region
    )

    return positive_coords, negative_coords


def sample_balanced(
    positive_coords,
    negative_coords
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n_positive = len(
        positive_coords
    )

    n_negative = min(
        len(negative_coords),
        n_positive
    )

    selected_negative = rng.choice(
        len(negative_coords),
        size=n_negative,
        replace=False
    )

    selected_negative = (
        negative_coords[
            selected_negative
        ]
    )

    coordinates = np.concatenate(
        [
            positive_coords,
            selected_negative
        ]
    )

    labels = np.concatenate(
        [
            np.ones(n_positive, dtype=np.uint8),
            np.zeros(n_negative, dtype=np.uint8)
        ]
    )

    permutation = rng.permutation(
        len(coordinates)
    )

    coordinates = coordinates[
        permutation
    ]

    labels = labels[
        permutation
    ]

    return coordinates, labels


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "\n========== LOADING ==========\n"
    )

    with rasterio.open(FEATURE_FILE) as features, \
         rasterio.open(LABEL_FILE) as labels:

        height = features.height
        width = features.width

        feature_mask = (
            features.read_masks(1) > 0
        )

        label_mask = (
            labels.read_masks(1) > 0
        )

        valid_mask = (
            feature_mask &
            label_mask
        )

        y = labels.read(1)

    print(
        "Raster:",
        height,
        "×",
        width
    )

    block_map = create_block_map(
        height,
        width
    )

    # --------------------------------------------------
    # TRAINING COORDINATES
    # --------------------------------------------------

    print(
        "\n========== TRAINING COORDINATES ==========\n"
    )

    train_positive, train_negative = \
        get_coordinates(
            y,
            valid_mask,
            block_map,
            0
        )

    print(
        "Positive:",
        len(train_positive)
    )

    print(
        "Negative:",
        len(train_negative)
    )

    train_coords, train_labels = \
        sample_balanced(
            train_positive,
            train_negative
        )

    # --------------------------------------------------
    # VALIDATION COORDINATES
    # --------------------------------------------------

    print(
        "\n========== VALIDATION COORDINATES ==========\n"
    )

    val_positive, val_negative = \
        get_coordinates(
            y,
            valid_mask,
            block_map,
            1
        )

    print(
        "Positive:",
        len(val_positive)
    )

    print(
        "Negative:",
        len(val_negative)
    )

    # Keep validation distribution natural.
    #
    # We do NOT balance it.

    val_coords = np.concatenate(
        [
            val_positive,
            val_negative
        ]
    )

    val_labels = np.concatenate(
        [
            np.ones(
                len(val_positive),
                dtype=np.uint8
            ),
            np.zeros(
                len(val_negative),
                dtype=np.uint8
            )
        ]
    )

    # Shuffle validation order only.
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    permutation = rng.permutation(
        len(val_coords)
    )

    val_coords = val_coords[
        permutation
    ]

    val_labels = val_labels[
        permutation
    ]

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    np.save(
        f"{OUTPUT_DIR}/train_coords.npy",
        train_coords
    )

    np.save(
        f"{OUTPUT_DIR}/train_labels.npy",
        train_labels
    )

    np.save(
        f"{OUTPUT_DIR}/val_coords.npy",
        val_coords
    )

    np.save(
        f"{OUTPUT_DIR}/val_labels.npy",
        val_labels
    )

    print(
        "\n========== SAVED ==========\n"
    )

    print(
        "Training:",
        train_coords.shape,
        train_labels.shape
    )

    print(
        "Validation:",
        val_coords.shape,
        val_labels.shape
    )

    print(
        "\nFiles saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()