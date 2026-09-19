import os

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_FILE = "data/raw/training_features.tif"
LABEL_FILE = "data/raw/Training_fault_labels.tif"

OUTPUT_DIR = "data/processed"

RANDOM_STATE = 42

# Number of negative samples relative to positive samples.
# 1.0 means approximately equal numbers.
NEGATIVE_TO_POSITIVE_RATIO = 1.0


def load_training_data():

    print("\n========== LOADING DATA ==========\n")

    with rasterio.open(FEATURE_FILE) as features, \
         rasterio.open(LABEL_FILE) as labels:

        feature_data = features.read()
        label_data = labels.read(1)

        feature_mask = features.read_masks(1) > 0
        label_mask = labels.read_masks(1) > 0

        valid_mask = feature_mask & label_mask

    print("Feature shape:", feature_data.shape)
    print("Label shape:", label_data.shape)

    # Convert from:
    #
    # (bands, height, width)
    #
    # to:
    #
    # (pixels, bands)

    X = feature_data[:, valid_mask].T

    y = label_data[valid_mask]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def remove_invalid_values(X, y):

    print("\n========== REMOVING INVALID VALUES ==========\n")

    valid = np.all(np.isfinite(X), axis=1)

    X = X[valid]
    y = y[valid]

    print("Remaining samples:", len(y))

    return X, y


def balance_dataset(X, y):

    print("\n========== BALANCING DATASET ==========\n")

    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]

    print("Positive samples:", len(positive_indices))
    print("Negative samples:", len(negative_indices))

    desired_negative_count = min(
        len(negative_indices),
        int(len(positive_indices) * NEGATIVE_TO_POSITIVE_RATIO)
    )

    rng = np.random.default_rng(RANDOM_STATE)

    selected_negative_indices = rng.choice(
        negative_indices,
        size=desired_negative_count,
        replace=False
    )

    selected_indices = np.concatenate([
        positive_indices,
        selected_negative_indices
    ])

    rng.shuffle(selected_indices)

    X = X[selected_indices]
    y = y[selected_indices]

    print("\nBalanced dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Fault:", np.sum(y == 1))
    print("Non-fault:", np.sum(y == 0))

    return X, y


def split_data(X, y):

    print("\n========== TRAIN / VALIDATION SPLIT ==========\n")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(y_train))
    print("Validation samples:", len(y_val))

    print("\nTraining faults:", np.sum(y_train == 1))
    print("Training non-faults:", np.sum(y_train == 0))

    print("\nValidation faults:", np.sum(y_val == 1))
    print("Validation non-faults:", np.sum(y_val == 0))

    return X_train, X_val, y_train, y_val


def scale_features(X_train, X_val):

    print("\n========== FEATURE SCALING ==========\n")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    return X_train, X_val, scaler


def save_data(X_train, X_val, y_train, y_val, scaler):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(
        f"{OUTPUT_DIR}/X_train.npy",
        X_train
    )

    np.save(
        f"{OUTPUT_DIR}/X_val.npy",
        X_val
    )

    np.save(
        f"{OUTPUT_DIR}/y_train.npy",
        y_train
    )

    np.save(
        f"{OUTPUT_DIR}/y_val.npy",
        y_val
    )

    # Save scaler parameters.
    np.save(
        f"{OUTPUT_DIR}/scaler_mean.npy",
        scaler.mean_
    )

    np.save(
        f"{OUTPUT_DIR}/scaler_scale.npy",
        scaler.scale_
    )

    print("\n========== SAVED ==========\n")

    print("Saved files to:", OUTPUT_DIR)


def main():

    X, y = load_training_data()

    X, y = remove_invalid_values(X, y)

    X, y = balance_dataset(X, y)

    X_train, X_val, y_train, y_val = split_data(X, y)

    X_train, X_val, scaler = scale_features(
        X_train,
        X_val
    )

    save_data(
        X_train,
        X_val,
        y_train,
        y_val,
        scaler
    )

    print("\n========== COMPLETE ==========\n")


if __name__ == "__main__":
    main()