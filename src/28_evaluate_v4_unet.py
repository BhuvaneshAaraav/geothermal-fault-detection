import os
import sys
import numpy as np
import torch
import torch.nn as nn
import rasterio
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"
LABELS_PATH = "data/raw/Training_fault_labels.tif"

MEANS_PATH = "data/processed/cnn/feature_means.npy"
STDS_PATH = "data/processed/cnn/feature_stds.npy"

MODEL_PATH = "models/fault_v4_unet_best.pt"

OUTPUT_DIR = "outputs"

PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    "v4_validation_probability.npy"
)

BINARY_PATH = os.path.join(
    OUTPUT_DIR,
    "v4_validation_predictions.npy"
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

PATCH_SIZE = 31
PATCH_RADIUS = 15

BATCH_SIZE = 16

ALPHA = 0.2
BETA = 0.8
DISTANCE_RADIUS = 3

EPS = 1e-7


# ============================================================
# U-NET
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class FaultSegmentationUNet(nn.Module):

    def __init__(self, in_channels=19):

        super().__init__()

        # Encoder

        self.enc1 = ConvBlock(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # Decoder

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)

        # Handle possible spatial mismatch
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = nn.functional.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = nn.functional.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING V4 U-NET")
    print("=" * 70)

    model = FaultSegmentationUNet(
        in_channels=19
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Handle either raw state_dict or checkpoint dictionary

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    print("Model loaded successfully:")
    print(MODEL_PATH)

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_rasters():

    print()
    print("=" * 70)
    print("LOADING RASTERS")
    print("=" * 70)

    with rasterio.open(FEATURES_PATH) as src:

        features = src.read().astype(
            np.float32
        )

    with rasterio.open(LABELS_PATH) as src:

        labels = src.read(1)

    print(
        "Features shape:",
        features.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    return features, labels


# ============================================================
# VALIDATION BLOCK MASK
# ============================================================

def create_validation_mask(height, width):

    """
    Same spatial split used throughout the project.

    4 x 4 blocks.

    Training:
        0,2,3,4,5,6,8,11,12,13,14,15

    Validation:
        1,7,9,10
    """

    rows_per_block = height // 4
    cols_per_block = width // 4

    validation_blocks = [
        1,
        7,
        9,
        10
    ]

    mask = np.zeros(
        (height, width),
        dtype=bool
    )

    for block_id in validation_blocks:

        block_row = block_id // 4
        block_col = block_id % 4

        r0 = block_row * rows_per_block
        c0 = block_col * cols_per_block

        if block_row == 3:
            r1 = height
        else:
            r1 = (block_row + 1) * rows_per_block

        if block_col == 3:
            c1 = width
        else:
            c1 = (block_col + 1) * cols_per_block

        mask[
            r0:r1,
            c0:c1
        ] = True

    return mask


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(features, means, stds):

    print()
    print("=" * 70)
    print("NORMALIZING FEATURES")
    print("=" * 70)

    normalized = np.empty_like(
        features,
        dtype=np.float32
    )

    invalid_mask = np.zeros(
        features.shape[1:],
        dtype=bool
    )

    for band in range(
        features.shape[0]
    ):

        x = features[band].copy()

        invalid = (
            ~np.isfinite(x)
            |
            (np.abs(x) > 1e30)
        )

        invalid_mask |= invalid

        # Replace nodata / invalid values
        # before normalization.

        x[invalid] = means[band]

        x = (
            x - means[band]
        ) / (
            stds[band] + 1e-8
        )

        # Prevent extreme normalized values.

        x = np.clip(
            x,
            -10.0,
            10.0
        )

        # Invalid pixels become zero
        # after normalization.

        x[invalid] = 0.0

        normalized[band] = x

    print(
        "Invalid pixels:",
        int(invalid_mask.sum())
    )

    print(
        "Normalized min:",
        float(normalized.min())
    )

    print(
        "Normalized max:",
        float(normalized.max())
    )

    return normalized, invalid_mask


# ============================================================
# SAFE VALIDATION MASK
# ============================================================

def create_safe_validation_mask(
    validation_mask,
    invalid_mask
):

    """
    Removes pixels too close to the raster boundary
    because inference uses 31x31 patches.
    """

    height, width = validation_mask.shape

    safe = validation_mask.copy()

    r = PATCH_RADIUS

    safe[:r, :] = False
    safe[-r:, :] = False
    safe[:, :r] = False
    safe[:, -r:] = False

    # Also exclude invalid feature pixels.

    safe &= ~invalid_mask

    return safe


# ============================================================
# FULL RASTER PATCH INFERENCE
# ============================================================

@torch.no_grad()
def predict_validation_raster(
    model,
    features,
    validation_mask,
    safe_mask
):

    print()
    print("=" * 70)
    print("FULL VALIDATION RASTER INFERENCE")
    print("=" * 70)

    height, width = validation_mask.shape

    probability = np.zeros(
        (height, width),
        dtype=np.float32
    )

    coords = np.argwhere(
        safe_mask
    )

    print(
        "Validation pixels:",
        int(validation_mask.sum())
    )

    print(
        "Safe validation pixels:",
        len(coords)
    )

    total_batches = (
        len(coords) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_idx in range(
        total_batches
    ):

        batch_coords = coords[
            batch_idx * BATCH_SIZE:
            (batch_idx + 1) * BATCH_SIZE
        ]

        patches = []

        for row, col in batch_coords:

            patch = features[
                :,
                row - PATCH_RADIUS:
                row + PATCH_RADIUS + 1,
                col - PATCH_RADIUS:
                col + PATCH_RADIUS + 1
            ]

            patches.append(patch)

        batch = np.stack(
            patches,
            axis=0
        )

        batch_tensor = torch.from_numpy(
            batch
        ).to(
            DEVICE,
            dtype=torch.float32
        )

        logits = model(
            batch_tensor
        )

        # Model output is [B,1,H,W].
        #
        # We use the CENTER prediction because
        # the competition produces one probability
        # per raster pixel.

        center = logits[
            :,
            0,
            PATCH_RADIUS,
            PATCH_RADIUS
        ]

        probs = torch.sigmoid(
            center
        ).cpu().numpy()

        for i, (row, col) in enumerate(
            batch_coords
        ):

            probability[
                row,
                col
            ] = probs[i]

        if (
            (batch_idx + 1) % 100 == 0
            or batch_idx == total_batches - 1
        ):

            print(
                f"Batch "
                f"{batch_idx + 1}/"
                f"{total_batches}"
            )

    return probability


# ============================================================
# COMPETITION METRIC
# ============================================================

def competition_dti(
    probability,
    labels,
    evaluation_mask,
    alpha=0.2,
    beta=0.8,
    radius=3
):

    """
    Distance-weighted Tversky metric.

    For each true fault pixel:

        TP contribution =
            max predicted probability
            within `radius`

    For false-positive background pixels:

        FP penalty is reduced when close
        to a true fault.

    This is the competition-style metric
    used in the project.

    """

    gt = (
        labels > 0
    )

    pred = np.asarray(
        probability,
        dtype=np.float64
    )

    eval_mask = (
        evaluation_mask
    )

    gt_eval = (
        gt & eval_mask
    )

    pred_eval = pred.copy()

    pred_eval[~eval_mask] = 0.0

    # --------------------------------------------------------
    # TRUE POSITIVE
    # --------------------------------------------------------

    distance_to_fault = distance_transform_edt(
        ~gt_eval
    )

    near_fault = (
        distance_to_fault <= radius
    )

    weighted_tp = 0.0

    fault_coords = np.argwhere(
        gt_eval
    )

    for row, col in fault_coords:

        r0 = max(
            0,
            row - radius
        )

        r1 = min(
            pred.shape[0],
            row + radius + 1
        )

        c0 = max(
            0,
            col - radius
        )

        c1 = min(
            pred.shape[1],
            col + radius + 1
        )

        local_pred = pred[
            r0:r1,
            c0:c1
        ]

        local_eval = eval_mask[
            r0:r1,
            c0:c1
        ]

        if np.any(local_eval):

            weighted_tp += np.max(
                local_pred[
                    local_eval
                ]
            )

    # --------------------------------------------------------
    # FALSE POSITIVE
    # --------------------------------------------------------

    background = (
        eval_mask
        &
        ~gt_eval
    )

    # Distance from each background pixel
    # to nearest true fault.

    distances = distance_transform_edt(
        ~gt_eval
    )

    # Pixels farther from faults receive
    # full FP penalty.
    #
    # Nearby predictions are discounted.

    fp_weight = np.ones_like(
        pred,
        dtype=np.float64
    )

    close = (
        distances <= radius
    )

    # Linear distance weighting:
    #
    # distance 0 -> weight 0
    # distance radius -> weight 1

    fp_weight[close] = (
        distances[close]
        / radius
    )

    weighted_fp = np.sum(
        pred_eval[
            background
        ]
        *
        fp_weight[
            background
        ]
    )

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    weighted_fn = 0.0

    for row, col in fault_coords:

        r0 = max(
            0,
            row - radius
        )

        r1 = min(
            pred.shape[0],
            row + radius + 1
        )

        c0 = max(
            0,
            col - radius
        )

        c1 = min(
            pred.shape[1],
            col + radius + 1
        )

        local_pred = pred[
            r0:r1,
            c0:c1
        ]

        local_eval = eval_mask[
            r0:r1,
            c0:c1
        ]

        if np.any(local_eval):

            best_near_fault = np.max(
                local_pred[
                    local_eval
                ]
            )

            weighted_fn += (
                1.0 - best_near_fault
            )

    # --------------------------------------------------------
    # TVERSKY
    # --------------------------------------------------------

    denominator = (
        weighted_tp
        +
        alpha * weighted_fp
        +
        beta * weighted_fn
        +
        EPS
    )

    score = (
        weighted_tp
        /
        denominator
    )

    return (
        float(score),
        float(weighted_tp),
        float(weighted_fp),
        float(weighted_fn)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    probability,
    labels,
    evaluation_mask
):

    print()
    print("=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)

    y_true = (
        labels[evaluation_mask] > 0
    ).astype(
        np.uint8
    )

    y_score = probability[
        evaluation_mask
    ]

    print(
        "Evaluation pixels:",
        len(y_true)
    )

    print(
        "Fault pixels:",
        int(y_true.sum())
    )

    print(
        "Background pixels:",
        int((y_true == 0).sum())
    )

    if (
        y_true.sum() > 0
        and
        y_true.sum() < len(y_true)
    ):

        roc = roc_auc_score(
            y_true,
            y_score
        )

        pr = average_precision_score(
            y_true,
            y_score
        )

    else:

        roc = float("nan")
        pr = float("nan")

    print(
        f"ROC-AUC : {roc:.6f}"
    )

    print(
        f"PR-AUC  : {pr:.6f}"
    )

    return roc, pr


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    probability,
    labels,
    evaluation_mask
):

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    eval_probs = probability[
        evaluation_mask
    ]

    eval_labels = (
        labels[evaluation_mask] > 0
    )

    thresholds = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]

    for threshold in thresholds:

        predicted = (
            eval_probs >= threshold
        )

        tp = np.sum(
            predicted & eval_labels
        )

        fp = np.sum(
            predicted & ~eval_labels
        )

        fn = np.sum(
            ~predicted & eval_labels
        )

        precision = (
            tp /
            (tp + fp + EPS)
        )

        recall = (
            tp /
            (tp + fn + EPS)
        )

        print(
            f"Threshold {threshold:.1f} | "
            f"Predicted {predicted.sum():8d} | "
            f"TP {tp:6d} | "
            f"FP {fp:8d} | "
            f"FN {fn:6d} | "
            f"Precision {precision:.5f} | "
            f"Recall {recall:.5f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V4 FULL VALIDATION EVALUATION")
    print("=" * 70)

    print(
        "Using device:",
        DEVICE
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD NORMALIZATION
    # --------------------------------------------------------

    print()
    print(
        "Loading normalization statistics..."
    )

    means = np.load(
        MEANS_PATH
    ).astype(
        np.float32
    )

    stds = np.load(
        STDS_PATH
    ).astype(
        np.float32
    )

    print(
        "Means:",
        means.shape
    )

    print(
        "Stds:",
        stds.shape
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    features, labels = load_rasters()

    height, width = labels.shape

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    features, invalid_mask = normalize_features(
        features,
        means,
        stds
    )

    # --------------------------------------------------------
    # VALIDATION MASK
    # --------------------------------------------------------

    validation_mask = create_validation_mask(
        height,
        width
    )

    safe_validation_mask = create_safe_validation_mask(
        validation_mask,
        invalid_mask
    )

    print()
    print(
        "Validation pixels:",
        int(validation_mask.sum())
    )

    print(
        "Safe validation pixels:",
        int(safe_validation_mask.sum())
    )

    validation_faults = (
        (labels > 0)
        &
        validation_mask
    )

    print(
        "Validation fault pixels:",
        int(validation_faults.sum())
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    probability = predict_validation_raster(
        model,
        features,
        validation_mask,
        safe_validation_mask
    )

    # Only evaluate safe pixels.

    probability[
        ~safe_validation_mask
    ] = 0.0

    # --------------------------------------------------------
    # SAVE PROBABILITY MAP
    # --------------------------------------------------------

    np.save(
        PREDICTION_PATH,
        probability
    )

    print()
    print(
        "Saved probability map:"
    )

    print(
        PREDICTION_PATH
    )

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    valid_probs = probability[
        safe_validation_mask
    ]

    print()
    print("=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    print(
        "Min    :",
        float(valid_probs.min())
    )

    print(
        "Max    :",
        float(valid_probs.max())
    )

    print(
        "Mean   :",
        float(valid_probs.mean())
    )

    print(
        "Median :",
        float(np.median(valid_probs))
    )

    for threshold in [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]:

        count = np.sum(
            valid_probs >= threshold
        )

        print(
            f">={threshold:.1f}:",
            int(count)
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    roc, pr = calculate_metrics(
        probability,
        labels,
        safe_validation_mask
    )

    # --------------------------------------------------------
    # COMPETITION DTI
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPETITION DISTANCE-WEIGHTED TVERSKY")
    print("=" * 70)

    dti, weighted_tp, weighted_fp, weighted_fn = (
        competition_dti(
            probability,
            labels,
            safe_validation_mask,
            alpha=ALPHA,
            beta=BETA,
            radius=DISTANCE_RADIUS
        )
    )

    print(
        f"DTI score       : {dti:.6f}"
    )

    print(
        f"Weighted TP     : {weighted_tp:.6f}"
    )

    print(
        f"Weighted FP     : {weighted_fp:.6f}"
    )

    print(
        f"Weighted FN     : {weighted_fn:.6f}"
    )

    # --------------------------------------------------------
    # BINARY MAP
    # --------------------------------------------------------

    # 0.5 is only for visualization.
    # It is NOT claimed to be the competition-optimal threshold.

    binary = (
        probability >= 0.5
    ).astype(
        np.uint8
    )

    binary[
        ~safe_validation_mask
    ] = 0

    np.save(
        BINARY_PATH,
        binary
    )

    print()
    print(
        "Saved binary map:"
    )

    print(
        BINARY_PATH
    )

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------

    threshold_analysis(
        probability,
        labels,
        safe_validation_mask
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"ROC-AUC : {roc:.6f}"
    )

    print(
        f"PR-AUC  : {pr:.6f}"
    )

    print(
        f"DTI     : {dti:.6f}"
    )

    print()
    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Probability:",
        PREDICTION_PATH
    )


if __name__ == "__main__":
    main()