import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


FEATURE_FILE = "data/raw/training_features.tif"
LABEL_FILE = "data/raw/Training_fault_labels.tif"

OUTPUT_DIR = "outputs"

N_BLOCK_ROWS = 4
N_BLOCK_COLS = 4

# These are the blocks selected by our previous script.
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

            if row == N_BLOCK_ROWS - 1:
                row_end = height
            else:
                row_end = (row + 1) * block_height

            if col == N_BLOCK_COLS - 1:
                col_end = width
            else:
                col_end = (col + 1) * block_width

            if block_id in TRAIN_BLOCKS:
                value = 0

            elif block_id in VALIDATION_BLOCKS:
                value = 1

            else:
                value = -1

            block_map[
                row_start:row_end,
                col_start:col_end
            ] = value

            block_id += 1

    return block_map


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n========== LOADING DATA ==========\n")

    with rasterio.open(FEATURE_FILE) as features:

        background = features.read(
            12,
            masked=True
        )

        height = features.height
        width = features.width

    with rasterio.open(LABEL_FILE) as labels:

        faults = labels.read(
            1,
            masked=True
        )

    print("Raster size:", height, "×", width)

    block_map = create_block_map(
        height,
        width
    )

    # --------------------------------------------------
    # Plot 1: Spatial split
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        background,
        cmap="gray"
    )

    # Validation blocks
    validation_mask = (
        block_map == 1
    )

    plt.imshow(
        np.ma.masked_where(
            ~validation_mask,
            validation_mask
        ),
        alpha=0.35
    )

    # Draw block boundaries
    block_height = height // N_BLOCK_ROWS
    block_width = width // N_BLOCK_COLS

    for r in range(1, N_BLOCK_ROWS):

        plt.axhline(
            r * block_height,
            linewidth=1
        )

    for c in range(1, N_BLOCK_COLS):

        plt.axvline(
            c * block_width,
            linewidth=1
        )

    plt.title(
        "Spatial Train / Validation Split"
    )

    plt.xlabel("Pixel column")
    plt.ylabel("Pixel row")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/spatial_split.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Plot 2: Fault map
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        background,
        cmap="gray"
    )

    fault_mask = (
        faults.filled(0) > 0
    )

    plt.imshow(
        np.ma.masked_where(
            ~fault_mask,
            fault_mask
        ),
        alpha=0.8
    )

    plt.title(
        "Known Fault Locations"
    )

    plt.xlabel("Pixel column")
    plt.ylabel("Pixel row")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/known_faults.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Plot 3: Faults + spatial split
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        background,
        cmap="gray"
    )

    # Faults
    plt.imshow(
        np.ma.masked_where(
            ~fault_mask,
            fault_mask
        ),
        alpha=0.75
    )

    # Validation blocks
    plt.imshow(
        np.ma.masked_where(
            ~validation_mask,
            validation_mask
        ),
        alpha=0.25
    )

    for r in range(1, N_BLOCK_ROWS):

        plt.axhline(
            r * block_height,
            linewidth=1
        )

    for c in range(1, N_BLOCK_COLS):

        plt.axvline(
            c * block_width,
            linewidth=1
        )

    plt.title(
        "Known Faults + Validation Regions"
    )

    plt.xlabel("Pixel column")
    plt.ylabel("Pixel row")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/faults_spatial_split.png",
        dpi=150
    )

    plt.close()

    print("\n========== SAVED ==========\n")

    print(
        "outputs/spatial_split.png"
    )

    print(
        "outputs/known_faults.png"
    )

    print(
        "outputs/faults_spatial_split.png"
    )


if __name__ == "__main__":
    main()