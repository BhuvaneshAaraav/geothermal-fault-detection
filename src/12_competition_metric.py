import numpy as np
from scipy.ndimage import maximum_filter


# ============================================================
# DISTANCE-WEIGHTED TVERSKY
# ============================================================

ALPHA = 0.2
BETA = 0.8

# Competition radius:
# 300 metres / 100 metres per pixel = 3 pixels
R = 3


def distance_weighted_tversky(
    prediction,
    ground_truth,
    alpha=ALPHA,
    beta=BETA,
    radius=R
):
    """
    Distance-weighted Tversky score.

    prediction:
        2D float array in [0, 1]

    ground_truth:
        2D binary array

    radius:
        Search radius in pixels.

    Triangular kernel:
        k(d) = max(1 - d/R, 0)
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    ground_truth = (
        np.asarray(ground_truth) > 0
    )

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Prediction and ground truth "
            "must have the same shape."
        )

    if prediction.ndim != 2:
        raise ValueError(
            "Inputs must be 2D arrays."
        )

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Build triangular kernel
    # --------------------------------------------------------

    size = 2 * radius + 1

    yy, xx = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    distance = np.sqrt(
        xx.astype(np.float64) ** 2 +
        yy.astype(np.float64) ** 2
    )

    kernel = np.maximum(
        1.0 - distance / radius,
        0.0
    )

    # --------------------------------------------------------
    # TP weighted by proximity to ground truth
    # --------------------------------------------------------

    #
    # For every ground-truth pixel g:
    #
    # max_x p(x) * k(d(x,g))
    #
    # We calculate this using local weighted windows.
    #

    gt_positions = np.argwhere(
        ground_truth
    )

    weighted_tp = 0.0

    weighted_fn = 0.0

    height, width = prediction.shape

    for gy, gx in gt_positions:

        y0 = max(
            0,
            gy - radius
        )

        y1 = min(
            height,
            gy + radius + 1
        )

        x0 = max(
            0,
            gx - radius
        )

        x1 = min(
            width,
            gx + radius + 1
        )

        ky0 = y0 - (gy - radius)
        ky1 = ky0 + (y1 - y0)

        kx0 = x0 - (gx - radius)
        kx1 = kx0 + (x1 - x0)

        local_prediction = prediction[
            y0:y1,
            x0:x1
        ]

        local_kernel = kernel[
            ky0:ky1,
            kx0:kx1
        ]

        local_score = np.max(
            local_prediction *
            local_kernel
        )

        weighted_tp += local_score

        weighted_fn += (
            1.0 - local_score
        )

    # --------------------------------------------------------
    # FP weighted by distance to ground truth
    # --------------------------------------------------------

    #
    # For each prediction pixel:
    #
    # p(x) * (1 - max_g k(d(x,g)))
    #
    # --------------------------------------------------------

    # If there are no known faults, every prediction is FP.

    if len(gt_positions) == 0:

        weighted_fp = float(
            prediction.sum()
        )

    else:

        # Create binary ground-truth map.

        gt_float = ground_truth.astype(
            np.float64
        )

        # Local maximum of the triangular influence.
        #
        # maximum_filter gives the nearest-window influence
        # when combined with the radial kernel approximation.
        #
        # We construct the exact weighted influence below.

        gt_influence = np.zeros_like(
            prediction,
            dtype=np.float64
        )

        # Iterate over ground truth pixels.
        #
        # This is intentionally simple and exact for the
        # competition radius of only 3 pixels.

        for gy, gx in gt_positions:

            y0 = max(
                0,
                gy - radius
            )

            y1 = min(
                height,
                gy + radius + 1
            )

            x0 = max(
                0,
                gx - radius
            )

            x1 = min(
                width,
                gx + radius + 1
            )

            ky0 = y0 - (gy - radius)
            ky1 = ky0 + (y1 - y0)

            kx0 = x0 - (gx - radius)
            kx1 = kx0 + (x1 - x0)

            gt_influence[
                y0:y1,
                x0:x1
            ] = np.maximum(
                gt_influence[
                    y0:y1,
                    x0:x1
                ],
                kernel[
                    ky0:ky1,
                    kx0:kx1
                ]
            )

        weighted_fp = np.sum(
            prediction *
            (1.0 - gt_influence)
        )

    # --------------------------------------------------------
    # Tversky
    # --------------------------------------------------------

    denominator = (
        weighted_tp
        + alpha * weighted_fp
        + beta * weighted_fn
    )

    if denominator <= 1e-12:
        return 1.0

    score = (
        weighted_tp /
        denominator
    )

    return float(score)


# ============================================================
# SIMPLE TESTS
# ============================================================

if __name__ == "__main__":

    print("\n========== TESTING COMPETITION METRIC ==========\n")

    # --------------------------------------------------------
    # Test 1: perfect prediction
    # --------------------------------------------------------

    gt = np.zeros(
        (20, 20),
        dtype=np.uint8
    )

    gt[10, 10] = 1

    prediction = gt.astype(
        np.float64
    )

    score = distance_weighted_tversky(
        prediction,
        gt
    )

    print(
        "Perfect prediction:",
        score
    )

    # --------------------------------------------------------
    # Test 2: prediction one pixel away
    # --------------------------------------------------------

    prediction = np.zeros(
        (20, 20),
        dtype=np.float64
    )

    prediction[10, 11] = 1.0

    score = distance_weighted_tversky(
        prediction,
        gt
    )

    print(
        "1-pixel displaced prediction:",
        score
    )

    # --------------------------------------------------------
    # Test 3: completely wrong location
    # --------------------------------------------------------

    prediction = np.zeros(
        (20, 20),
        dtype=np.float64
    )

    prediction[0, 0] = 1.0

    score = distance_weighted_tversky(
        prediction,
        gt
    )

    print(
        "Far-away prediction:",
        score
    )

    # --------------------------------------------------------
    # Test 4: no prediction
    # --------------------------------------------------------

    prediction = np.zeros(
        (20, 20),
        dtype=np.float64
    )

    score = distance_weighted_tversky(
        prediction,
        gt
    )

    print(
        "No prediction:",
        score
    )

    # --------------------------------------------------------
    # Test 5: soft prediction
    # --------------------------------------------------------

    prediction = np.zeros(
        (20, 20),
        dtype=np.float64
    )

    prediction[10, 10] = 0.7
    prediction[10, 11] = 0.5
    prediction[10, 12] = 0.3

    score = distance_weighted_tversky(
        prediction,
        gt
    )

    print(
        "Soft prediction:",
        score
    )

    print("\n========== DONE ==========\n")