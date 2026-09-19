import os
import numpy as np


TRAIN_FILE = "data/processed/spatial/X_train.npy"

OUTPUT_DIR = "data/processed/cnn"

FEATURE_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
    "deq_n100a15",
    "iso_grav_anom_vg",
    "det_elev",
    "iso_grav_anom",
    "tmi",
    "depth_to_base_surf",
    "ieq_n100a15",
    "cond_surf",
    "iso_grav_anom_hg",
    "det_elev_slope"
]


def main():

    print("\n========== LOADING TRAINING DATA ==========\n")

    X_train = np.load(
        TRAIN_FILE
    ).astype(
        np.float64
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "\n========== CALCULATING STATISTICS ==========\n"
    )

    means = np.zeros(
        X_train.shape[1],
        dtype=np.float64
    )

    stds = np.zeros(
        X_train.shape[1],
        dtype=np.float64
    )

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        values = X_train[:, i]

        finite = np.isfinite(
            values
        )

        values = values[finite]

        means[i] = np.mean(
            values,
            dtype=np.float64
        )

        stds[i] = np.std(
            values,
            dtype=np.float64
        )

        # Prevent division by zero.
        if stds[i] < 1e-8:
            stds[i] = 1.0

        print(
            f"{name:25s} "
            f"mean={means[i]:12.4f} "
            f"std={stds[i]:12.4f}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.save(
        f"{OUTPUT_DIR}/feature_means.npy",
        means
    )

    np.save(
        f"{OUTPUT_DIR}/feature_stds.npy",
        stds
    )

    print(
        "\n========== SAVED ==========\n"
    )

    print(
        f"{OUTPUT_DIR}/feature_means.npy"
    )

    print(
        f"{OUTPUT_DIR}/feature_stds.npy"
    )


if __name__ == "__main__":
    main()