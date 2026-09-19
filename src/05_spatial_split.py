import os

import numpy as np
import rasterio


FEATURE_FILE = "data/raw/training_features.tif"
LABEL_FILE = "data/raw/Training_fault_labels.tif"

OUTPUT_DIR = "data/processed/spatial"

RANDOM_STATE = 42

# Number of geographic blocks.
# The raster is divided into this many blocks in each direction.
N_BLOCK_ROWS = 4
N_BLOCK_COLS = 4

# Percentage of blocks used for validation.
VALIDATION_BLOCK_FRACTION = 0.25

# Number of negative pixels to sample for every positive pixel.
NEGATIVE_TO_POSITIVE_RATIO = 1.0


def load_data():

    print("\n========== LOADING RASTER ==========\n")

    with rasterio.open(FEATURE_FILE) as features, \
         rasterio.open(LABEL_FILE) as labels:

        X = features.read()
        y = labels.read(1)

        feature_mask = features.read_masks(1) > 0
        label_mask = labels.read_masks(1) > 0

        valid_mask = feature_mask & label_mask

    print("Features:", X.shape)
    print("Labels:", y.shape)

    return X, y, valid_mask


def create_spatial_blocks(height, width):

    print("\n========== CREATING SPATIAL BLOCKS ==========\n")

    block_height = height // N_BLOCK_ROWS
    block_width = width // N_BLOCK_COLS

    blocks = []

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

            blocks.append(
                {
                    "id": row * N_BLOCK_COLS + col,
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end
                }
            )

    for block in blocks:

        print(
            f"Block {block['id']:2d}: "
            f"rows {block['row_start']}:{block['row_end']} "
            f"cols {block['col_start']}:{block['col_end']}"
        )

    return blocks


def choose_validation_blocks(blocks):

    rng = np.random.default_rng(RANDOM_STATE)

    number_of_validation_blocks = max(
        1,
        int(len(blocks) * VALIDATION_BLOCK_FRACTION)
    )

    validation_ids = rng.choice(
        len(blocks),
        size=number_of_validation_blocks,
        replace=False
    )

    validation_ids = set(validation_ids)

    train_blocks = [
        block for i, block in enumerate(blocks)
        if i not in validation_ids
    ]

    validation_blocks = [
        block for i, block in enumerate(blocks)
        if i in validation_ids
    ]

    print("\nTraining blocks:")

    print([
        block["id"]
        for block in train_blocks
    ])

    print("\nValidation blocks:")

    print([
        block["id"]
        for block in validation_blocks
    ])

    return train_blocks, validation_blocks


def extract_block_samples(X, y, valid_mask, blocks):

    all_features = []
    all_labels = []

    for block in blocks:

        rows = slice(
            block["row_start"],
            block["row_end"]
        )

        cols = slice(
            block["col_start"],
            block["col_end"]
        )

        block_X = X[:, rows, cols]
        block_y = y[rows, cols]
        block_valid = valid_mask[rows, cols]

        # Convert:
        #
        # (bands, rows, cols)
        #
        # into:
        #
        # (pixels, bands)

        block_X = block_X[:, block_valid].T
        block_y = block_y[block_valid]

        # Remove NaN / infinite values

        finite = np.all(
            np.isfinite(block_X),
            axis=1
        )

        block_X = block_X[finite]
        block_y = block_y[finite]

        all_features.append(block_X)
        all_labels.append(block_y)

    X_result = np.concatenate(
        all_features,
        axis=0
    )

    y_result = np.concatenate(
        all_labels,
        axis=0
    )

    return X_result, y_result


def balance_training_data(X, y):

    print("\n========== BALANCING TRAINING DATA ==========\n")

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    print("Positive:", len(positive))
    print("Negative:", len(negative))

    desired_negative = min(
        len(negative),
        int(
            len(positive)
            * NEGATIVE_TO_POSITIVE_RATIO
        )
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    selected_negative = rng.choice(
        negative,
        size=desired_negative,
        replace=False
    )

    selected = np.concatenate(
        [
            positive,
            selected_negative
        ]
    )

    rng.shuffle(selected)

    return X[selected], y[selected]


def save_data(
    X_train,
    y_train,
    X_val,
    y_val
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.save(
        f"{OUTPUT_DIR}/X_train.npy",
        X_train
    )

    np.save(
        f"{OUTPUT_DIR}/y_train.npy",
        y_train
    )

    np.save(
        f"{OUTPUT_DIR}/X_val.npy",
        X_val
    )

    np.save(
        f"{OUTPUT_DIR}/y_val.npy",
        y_val
    )

    print("\n========== SAVED ==========\n")

    print(
        f"Saved to {OUTPUT_DIR}/"
    )


def main():

    X, y, valid_mask = load_data()

    height = y.shape[0]
    width = y.shape[1]

    blocks = create_spatial_blocks(
        height,
        width
    )

    train_blocks, validation_blocks = \
        choose_validation_blocks(blocks)

    print("\n========== EXTRACTING TRAINING BLOCKS ==========\n")

    X_train, y_train = extract_block_samples(
        X,
        y,
        valid_mask,
        train_blocks
    )

    print(
        "Raw training:",
        X_train.shape,
        y_train.shape
    )

    print("\n========== EXTRACTING VALIDATION BLOCKS ==========\n")

    X_val, y_val = extract_block_samples(
        X,
        y,
        valid_mask,
        validation_blocks
    )

    print(
        "Raw validation:",
        X_val.shape,
        y_val.shape
    )

    # Balance ONLY the training data.
    #
    # Validation must remain untouched.

    X_train, y_train = balance_training_data(
        X_train,
        y_train
    )

    print("\nFinal training:")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    print("\nValidation:")
    print("X:", X_val.shape)
    print("y:", y_val.shape)

    save_data(
        X_train,
        y_train,
        X_val,
        y_val
    )


if __name__ == "__main__":
    main()