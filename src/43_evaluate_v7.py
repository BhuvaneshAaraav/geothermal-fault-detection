import os
import numpy as np
import rasterio
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PATCH_SIZE = 31
RADIUS = PATCH_SIZE // 2


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = "data/raw/training_features.tif"
LABEL_PATH = "data/raw/Training_fault_labels.tif"

VAL_COORDS_PATH = (
    "data/processed/unet/val_coords.npy"
)

MEANS_PATH = (
    "data/processed/cnn/feature_means.npy"
)

STDS_PATH = (
    "data/processed/cnn/feature_stds.npy"
)

MODEL_PATH = (
    "models/fault_v7_distance_unet_best.pt"
)

OUTPUT_PATH = (
    "outputs/v7_validation_probability.npy"
)


# ============================================================
# MODEL
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
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class FaultUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = ConvBlock(
            19,
            32
        )

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = ConvBlock(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

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
# START
# ============================================================

print("=" * 70)
print("V7 VALIDATION INFERENCE")
print("=" * 70)

print("\nUsing device:", DEVICE)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading features...")

with rasterio.open(FEATURE_PATH) as src:

    features = src.read().astype(
        np.float32
    )

H = features.shape[1]
W = features.shape[2]

print(
    "Feature shape:",
    features.shape
)


# ============================================================
# LOAD LABELS
# ============================================================

with rasterio.open(LABEL_PATH) as src:

    labels = src.read(1)

print(
    "Label shape:",
    labels.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

means = np.load(
    MEANS_PATH
).astype(np.float32)

stds = np.load(
    STDS_PATH
).astype(np.float32
)

print("\nNormalizing...")

for b in range(
    features.shape[0]
):

    x = features[b]

    invalid = (
        ~np.isfinite(x)
        |
        (np.abs(x) > 1e30)
    )

    x[invalid] = means[b]

    x = (
        x - means[b]
    ) / (
        stds[b] + 1e-8
    )

    x = np.clip(
        x,
        -10.0,
        10.0
    )

    x[invalid] = 0.0

    features[b] = x


# ============================================================
# VALIDATION COORDINATES
# ============================================================

val_coords = np.load(
    VAL_COORDS_PATH
)

print(
    "\nValidation coordinates:",
    val_coords.shape
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading V7 model...")

model = FaultUNet().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded epoch:",
    checkpoint["epoch"]
)

print(
    "Training loss:",
    checkpoint["loss"]
)


# ============================================================
# FULL VALIDATION PROBABILITY MAP
# ============================================================

probability_map = np.zeros(
    (H, W),
    dtype=np.float32
)

ys = val_coords[:, 0]
xs = val_coords[:, 1]

total = len(val_coords)

BATCH_SIZE = 64

print("\nRunning inference...")

with torch.no_grad():

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_coords = val_coords[
            start:end
        ]

        patches = []

        valid_indices = []

        for i, (
            y,
            x
        ) in enumerate(
            batch_coords
        ):

            y = int(y)
            x = int(x)

            patch = features[
                :,
                y-RADIUS:y+RADIUS+1,
                x-RADIUS:x+RADIUS+1
            ]

            if patch.shape == (
                19,
                PATCH_SIZE,
                PATCH_SIZE
            ):

                patches.append(
                    patch
                )

                valid_indices.append(i)

        if not patches:
            continue

        batch = torch.from_numpy(
            np.stack(patches)
        ).to(DEVICE)

        logits = model(
            batch
        )

        probs = torch.sigmoid(
            logits
        )

        probs = probs.cpu().numpy()[
            :, 0
        ]

        # ----------------------------------------------------
        # Use CENTER prediction
        # ----------------------------------------------------

        center = probs[
            :,
            RADIUS,
            RADIUS
        ]

        for j, local_idx in enumerate(
            valid_indices
        ):

            y = int(
                batch_coords[
                    local_idx,
                    0
                ]
            )

            x = int(
                batch_coords[
                    local_idx,
                    1
                ]
            )

            probability_map[
                y,
                x
            ] = center[j]

        if (
            start // BATCH_SIZE
        ) % 100 == 0:

            print(
                f"Processed "
                f"{end:,}/{total:,}"
            )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    probability_map
)

print(
    "\nSaved:",
    OUTPUT_PATH
)


# ============================================================
# DISTRIBUTION
# ============================================================

validation_values = probability_map[
    ys,
    xs
]

print("\n" + "=" * 70)
print("V7 PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    "Min   :",
    float(validation_values.min())
)

print(
    "Median:",
    float(np.median(validation_values))
)

print(
    "Mean  :",
    float(validation_values.mean())
)

print(
    "P90   :",
    float(np.percentile(
        validation_values,
        90
    ))
)

print(
    "P99   :",
    float(np.percentile(
        validation_values,
        99
    ))
)

print(
    "Max   :",
    float(validation_values.max())
)

print(
    "\n>= 0.5:",
    int(
        np.sum(
            validation_values >= 0.5
        )
    )
)

print(
    ">= 0.7:",
    int(
        np.sum(
            validation_values >= 0.7
        )
    )
)

print(
    ">= 0.9:",
    int(
        np.sum(
            validation_values >= 0.9
        )
    )
)

print("\nDONE")