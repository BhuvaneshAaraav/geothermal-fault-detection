import numpy as np


# ============================================================
# CONFIG
# ============================================================

ALPHA = 0.2
BETA = 0.8
RADIUS = 3.0
EPS = 1e-12


# ============================================================
# REFERENCE IMPLEMENTATION
# ============================================================

def reference_metric(y_true, y_pred):
    """
    Direct/reference implementation.

    y_true: binary [H, W]
    y_pred: probability [H, W], values in [0, 1]
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    gt_positions = np.argwhere(y_true > 0)

    if len(gt_positions) == 0:
        return 1.0 if np.sum(y_pred) == 0 else 0.0

    tp_w = 0.0
    fn_w = 0.0

    H, W = y_true.shape

    for gy, gx in gt_positions:

        best = 0.0

        y0 = max(0, int(gy - RADIUS))
        y1 = min(H, int(gy + RADIUS) + 1)

        x0 = max(0, int(gx - RADIUS))
        x1 = min(W, int(gx + RADIUS) + 1)

        for py in range(y0, y1):
            for px in range(x0, x1):

                dy = py - gy
                dx = px - gx
                distance = np.sqrt(dy * dy + dx * dx)

                if distance <= RADIUS:

                    kernel = max(1.0 - distance / RADIUS, 0.0)

                    value = y_pred[py, px] * kernel

                    if value > best:
                        best = value

        tp_w += best
        fn_w += 1.0 - best

    # --------------------------------------------------------
    # False-positive contribution
    # --------------------------------------------------------

    fp_w = 0.0

    pred_positions = np.argwhere(y_pred > 0)

    for py, px in pred_positions:

        best_kernel = 0.0

        y0 = max(0, int(py - RADIUS))
        y1 = min(H, int(py + RADIUS) + 1)

        x0 = max(0, int(px - RADIUS))
        x1 = min(W, int(px + RADIUS) + 1)

        for gy, gx in gt_positions:

            dy = gy - py
            dx = gx - px
            distance = np.sqrt(dy * dy + dx * dx)

            if distance <= RADIUS:

                kernel = max(1.0 - distance / RADIUS, 0.0)

                if kernel > best_kernel:
                    best_kernel = kernel

        fp_w += y_pred[py, px] * (1.0 - best_kernel)

    denominator = (
        tp_w
        + ALPHA * fp_w
        + BETA * fn_w
        + EPS
    )

    return tp_w / denominator


# ============================================================
# FAST IMPLEMENTATION
# ============================================================

def fast_metric(y_true, y_pred):
    """
    Fast implementation using 49 local shifts.

    This should mathematically match the reference implementation.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    H, W = y_true.shape

    # --------------------------------------------------------
    # TP / FN
    #
    # For every GT pixel:
    #
    # max[pred(x) * kernel(distance)]
    #
    # --------------------------------------------------------

    best_match = np.zeros_like(y_pred)

    offsets = []

    max_offset = int(np.ceil(RADIUS))

    for dy in range(-max_offset, max_offset + 1):
        for dx in range(-max_offset, max_offset + 1):

            distance = np.sqrt(dy * dy + dx * dx)

            if distance <= RADIUS:

                kernel = max(
                    1.0 - distance / RADIUS,
                    0.0
                )

                offsets.append((dy, dx, kernel))

    for dy, dx, kernel in offsets:

        shifted = np.zeros_like(y_pred)

        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)

        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)

        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)

        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)

        shifted[
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = y_pred[
            src_y0:src_y1,
            src_x0:src_x1
        ] * kernel

        best_match = np.maximum(
            best_match,
            shifted
        )

    tp_w = np.sum(
        y_true * best_match
    )

    fn_w = np.sum(
        y_true * (1.0 - best_match)
    )

    # --------------------------------------------------------
    # FP
    #
    # For every prediction pixel:
    #
    # max[GT(x) * kernel(distance)]
    #
    # --------------------------------------------------------

    gt_match = np.zeros_like(y_true)

    for dy, dx, kernel in offsets:

        shifted = np.zeros_like(y_true)

        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)

        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)

        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)

        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)

        shifted[
            dst_y0:dst_y1,
            dst_x0:dst_x1
        ] = y_true[
            src_y0:src_y1,
            src_x0:src_x1
        ] * kernel

        gt_match = np.maximum(
            gt_match,
            shifted
        )

    fp_w = np.sum(
        y_pred * (1.0 - gt_match)
    )

    denominator = (
        tp_w
        + ALPHA * fp_w
        + BETA * fn_w
        + EPS
    )

    return tp_w / denominator


# ============================================================
# TEST HELPER
# ============================================================

def compare_case(name, y_true, y_pred):

    ref = reference_metric(
        y_true,
        y_pred
    )

    fast = fast_metric(
        y_true,
        y_pred
    )

    difference = abs(ref - fast)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Reference : {ref:.15f}")
    print(f"Fast      : {fast:.15f}")
    print(f"Difference: {difference:.15e}")

    if difference < 1e-10:
        print("PASS")
    else:
        print("FAIL")

    return difference


# ============================================================
# TESTS
# ============================================================

def main():

    print("=" * 70)
    print("VERIFYING COMPETITION METRIC")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Perfect prediction
    # --------------------------------------------------------

    y_true = np.zeros((15, 15), dtype=np.float64)
    y_true[7, 7] = 1.0

    y_pred = y_true.copy()

    compare_case(
        "TEST 1 — Perfect prediction",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 2. One-pixel displacement
    # --------------------------------------------------------

    y_true = np.zeros((15, 15), dtype=np.float64)
    y_true[7, 7] = 1.0

    y_pred = np.zeros_like(y_true)
    y_pred[7, 8] = 1.0

    compare_case(
        "TEST 2 — One-pixel displacement",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 3. Far-away prediction
    # --------------------------------------------------------

    y_true = np.zeros((20, 20), dtype=np.float64)
    y_true[5, 5] = 1.0

    y_pred = np.zeros_like(y_true)
    y_pred[15, 15] = 1.0

    compare_case(
        "TEST 3 — Far-away prediction",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 4. Soft nearby prediction
    # --------------------------------------------------------

    y_true = np.zeros((15, 15), dtype=np.float64)
    y_true[7, 7] = 1.0

    y_pred = np.zeros_like(y_true)

    y_pred[7, 7] = 0.8
    y_pred[7, 8] = 0.6
    y_pred[8, 7] = 0.4

    compare_case(
        "TEST 4 — Soft nearby prediction",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 5. No prediction
    # --------------------------------------------------------

    y_true = np.zeros((15, 15), dtype=np.float64)
    y_true[7, 7] = 1.0

    y_pred = np.zeros_like(y_true)

    compare_case(
        "TEST 5 — No prediction",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 6. Multiple faults
    # --------------------------------------------------------

    y_true = np.zeros((30, 30), dtype=np.float64)

    y_true[5, 5] = 1
    y_true[10, 20] = 1
    y_true[22, 10] = 1
    y_true[25, 25] = 1

    y_pred = np.zeros_like(y_true)

    y_pred[5, 6] = 0.9
    y_pred[10, 20] = 0.8
    y_pred[21, 10] = 0.7
    y_pred[27, 27] = 0.6

    compare_case(
        "TEST 6 — Multiple faults",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 7. Random test
    # --------------------------------------------------------

    np.random.seed(42)

    y_true = (
        np.random.random((20, 20)) < 0.03
    ).astype(np.float64)

    y_pred = np.random.random(
        (20, 20)
    ).astype(np.float64)

    compare_case(
        "TEST 7 — Random probability map",
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPECTED REFERENCE VALUES FROM EARLIER TESTS")
    print("=" * 70)

    print("Perfect prediction       ≈ 1.000000")
    print("1-pixel displacement     ≈ 0.666667")
    print("Far-away prediction      ≈ 0.000000")
    print("Soft nearby prediction   ≈ 0.690789")
    print("No prediction            ≈ 0.000000")

    print("\nIf every test says PASS, the fast implementation")
    print("is consistent with the reference implementation.")


if __name__ == "__main__":
    main()