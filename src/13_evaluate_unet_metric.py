import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"

DATA_DIR = "data/processed/unet"
STATS_DIR = "data/processed/cnn"

MODEL_PATH = "models/fault_unet_spatial_best.pt"

PATCH_SIZE = 31
BATCH_SIZE = 32

MAX_VAL_SAMPLES = 40000

SEED = 42

ALPHA = 0.2
BETA = 0.8
RADIUS = 3


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n========== LOADING VALIDATION DATA ==========\n")

val_coords = np.load(
    os.path.join(DATA_DIR, "val_coords.npy")
)

val_labels = np.load(
    os.path.join(DATA_DIR, "val_labels.npy")
)

print(
    "Full validation coordinates:",
    val_coords.shape
)


# Same deterministic validation subset used during training
if len(val_coords) > MAX_VAL_SAMPLES:

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(val_coords),
        size=MAX_VAL_SAMPLES,
        replace=False
    )

    val_coords = val_coords[indices]
    val_labels = val_labels[indices]


print(
    "Evaluation coordinates:",
    val_coords.shape
)

print(
    "Fault pixels:",
    int(val_labels.sum())
)

print(
    "Non-fault pixels:",
    int((val_labels == 0).sum())
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n========== LOADING NORMALIZATION ==========\n")

means = np.load(
    os.path.join(
        STATS_DIR,
        "feature_means.npy"
    )
).astype(np.float64)

stds = np.load(
    os.path.join(
        STATS_DIR,
        "feature_stds.npy"
    )
).astype(np.float64)

stds[stds < 1e-8] = 1.0

print("Means:", means.shape)
print("Stds :", stds.shape)


# ============================================================
# DATASET
# ============================================================

class GeoDataset(Dataset):

    def __init__(
        self,
        coords,
        labels,
        raster_path,
        means,
        stds,
        patch_size
    ):

        self.coords = coords
        self.labels = labels

        self.means = means
        self.stds = stds

        self.patch_size = patch_size
        self.radius = patch_size // 2

        self.src = rasterio.open(
            raster_path
        )

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):

        y, x = self.coords[index]

        y = int(y)
        x = int(x)

        r = self.radius

        window = rasterio.windows.Window(
            x - r,
            y - r,
            self.patch_size,
            self.patch_size
        )

        patch = self.src.read(
            window=window
        ).astype(np.float64)

        mask = self.src.read_masks(
            1,
            window=window
        )

        invalid = mask == 0

        for band_index in range(
            patch.shape[0]
        ):

            band = patch[band_index]

            bad = (
                invalid |
                ~np.isfinite(band)
            )

            band[bad] = self.means[band_index]

            patch[band_index] = band

        patch = (
            patch -
            self.means[:, None, None]
        ) / self.stds[:, None, None]

        patch = np.clip(
            patch,
            -10.0,
            10.0
        )

        patch = patch.astype(
            np.float32
        )

        return (
            torch.from_numpy(patch),
            torch.tensor(
                self.labels[index],
                dtype=torch.float32
            )
        )


# ============================================================
# DATASET / LOADER
# ============================================================

print("\n========== CREATING DATASET ==========\n")

dataset = GeoDataset(
    val_coords,
    val_labels,
    FEATURES_PATH,
    means,
    stds,
    PATCH_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# MODEL
# IMPORTANT:
# This architecture MUST match src/11_train_unet.py
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


class FaultUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------- Encoder ----------------

        self.enc1 = ConvBlock(
            19,
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

        # ---------------- Bottleneck ----------------

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # ---------------- Decoder ----------------

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

        # ---------------- Output ----------------

        self.final_conv = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        # THESE WERE MISSING IN THE PREVIOUS
        # EVALUATION SCRIPT.

        self.center_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                1,
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # ---------------- Encoder ----------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # ---------------- Bottleneck ----------------

        b = self.bottleneck(
            self.pool3(e3)
        )

        # ---------------- Decoder 3 ----------------

        d3 = self.up3(b)

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

        # ---------------- Decoder 2 ----------------

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

        # ---------------- Decoder 1 ----------------

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

        # ---------------- Feature map ----------------

        logits_map = self.final_conv(d1)

        # ---------------- Center 5×5 ----------------

        h = logits_map.shape[-2]
        w = logits_map.shape[-1]

        cy = h // 2
        cx = w // 2

        center_logits = logits_map[
            :,
            :,
            cy - 2:cy + 3,
            cx - 2:cx + 3
        ]

        center_logits = center_logits.mean(
            dim=(2, 3)
        )

        # IMPORTANT:
        # The training model returns center_logits.
        #
        # The classifier exists in the saved checkpoint,
        # but it is not actually used in forward().
        #
        # Therefore we must NOT use classifier here.

        return center_logits


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print(
    "\n========== LOADING BEST U-NET ==========\n"
)

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


model = FaultUNet().to(
    device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Loaded:",
    MODEL_PATH
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print(
    "\n========== GENERATING PREDICTIONS ==========\n"
)

predictions = []
targets = []


with torch.no_grad():

    for batch_index, (
        patches,
        labels
    ) in enumerate(loader):

        patches = patches.to(
            device
        )

        logits = model(
            patches
        )

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1)

        predictions.append(
            probabilities.cpu().numpy()
        )

        targets.append(
            labels.numpy()
        )

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(loader)} batches"
            )


predictions = np.concatenate(
    predictions
)

targets = np.concatenate(
    targets
)


print(
    "\nPrediction shape:",
    predictions.shape
)

print(
    "Prediction minimum:",
    float(predictions.min())
)

print(
    "Prediction maximum:",
    float(predictions.max())
)


# ============================================================
# ROC / PR AUC
# ============================================================

print(
    "\n========== CLASSIFICATION METRICS ==========\n"
)

roc_auc = roc_auc_score(
    targets,
    predictions
)

pr_auc = average_precision_score(
    targets,
    predictions
)

print(
    f"ROC-AUC: {roc_auc:.6f}"
)

print(
    f"PR-AUC : {pr_auc:.6f}"
)


# ============================================================
# RECONSTRUCT VALIDATION RASTER
# ============================================================

print(
    "\n========== RECONSTRUCTING VALIDATION RASTER ==========\n"
)

with rasterio.open(
    FEATURES_PATH
) as src:

    height = src.height
    width = src.width


prediction_raster = np.zeros(
    (height, width),
    dtype=np.float64
)

ground_truth_raster = np.zeros(
    (height, width),
    dtype=np.uint8
)


for i in range(
    len(val_coords)
):

    y = int(
        val_coords[i, 0]
    )

    x = int(
        val_coords[i, 1]
    )

    prediction_raster[
        y,
        x
    ] = predictions[i]

    ground_truth_raster[
        y,
        x
    ] = int(
        targets[i]
    )


# ============================================================
# DISTANCE-WEIGHTED TVERSKY
# ============================================================

def distance_weighted_tversky(
    prediction,
    ground_truth,
    alpha=0.2,
    beta=0.8,
    radius=3
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    ground_truth = (
        np.asarray(
            ground_truth
        ) > 0
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

    height, width = prediction.shape

    # --------------------------------------------------------
    # TRIANGULAR KERNEL
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    distance = np.sqrt(
        xx.astype(np.float64) ** 2
        +
        yy.astype(np.float64) ** 2
    )

    kernel = np.maximum(
        1.0 -
        distance / radius,
        0.0
    )

    # --------------------------------------------------------
    # GROUND TRUTH POSITIONS
    # --------------------------------------------------------

    gt_positions = np.argwhere(
        ground_truth
    )

    # --------------------------------------------------------
    # TP / FN
    # --------------------------------------------------------

    weighted_tp = 0.0
    weighted_fn = 0.0

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

        ky0 = (
            y0 -
            (gy - radius)
        )

        ky1 = (
            ky0 +
            (y1 - y0)
        )

        kx0 = (
            x0 -
            (gx - radius)
        )

        kx1 = (
            kx0 +
            (x1 - x0)
        )

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
            1.0 -
            local_score
        )

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    ground_truth_influence = np.zeros(
        prediction.shape,
        dtype=np.float64
    )

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

        ky0 = (
            y0 -
            (gy - radius)
        )

        ky1 = (
            ky0 +
            (y1 - y0)
        )

        kx0 = (
            x0 -
            (gx - radius)
        )

        kx1 = (
            kx0 +
            (x1 - x0)
        )

        ground_truth_influence[
            y0:y1,
            x0:x1
        ] = np.maximum(
            ground_truth_influence[
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
        (
            1.0 -
            ground_truth_influence
        )
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
    )

    if denominator <= 1e-12:
        return 1.0

    return (
        weighted_tp /
        denominator
    )


# ============================================================
# COMPETITION SCORE
# ============================================================

print(
    "\n========== COMPETITION-STYLE METRIC ==========\n"
)

score = distance_weighted_tversky(
    prediction_raster,
    ground_truth_raster,
    alpha=ALPHA,
    beta=BETA,
    radius=RADIUS
)

print(
    f"Distance-weighted Tversky: "
    f"{score:.6f}"
)


# ============================================================
# ZERO BASELINE
# ============================================================

zero_prediction = np.zeros_like(
    ground_truth_raster,
    dtype=np.float64
)

zero_score = distance_weighted_tversky(
    zero_prediction,
    ground_truth_raster,
    alpha=ALPHA,
    beta=BETA,
    radius=RADIUS
)

print(
    f"Zero prediction: "
    f"{zero_score:.6f}"
)


# ============================================================
# PERFECT BASELINE
# ============================================================

perfect_prediction = (
    ground_truth_raster.astype(
        np.float64
    )
)

perfect_score = distance_weighted_tversky(
    perfect_prediction,
    ground_truth_raster,
    alpha=ALPHA,
    beta=BETA,
    radius=RADIUS
)

print(
    f"Perfect prediction: "
    f"{perfect_score:.6f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "unet_validation_predictions.npy"
)

np.save(
    prediction_path,
    prediction_raster.astype(
        np.float32
    )
)

print(
    "\nSaved validation predictions:"
)

print(
    prediction_path
)


# ============================================================
# DONE
# ============================================================

print(
    "\n========== DONE ==========\n"
)