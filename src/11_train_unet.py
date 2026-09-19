import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = "data/raw/training_features.tif"

DATA_DIR = "data/processed/unet"
STATS_DIR = "data/processed/cnn"
MODEL_DIR = "models"

PATCH_SIZE = 31
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

# Keep validation manageable on the 8 GB M1.
MAX_VAL_SAMPLES = 40000

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


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
# LOAD DATA
# ============================================================

print("\n========== LOADING COORDINATES ==========\n")

train_coords = np.load(
    os.path.join(DATA_DIR, "train_coords.npy")
)

train_labels = np.load(
    os.path.join(DATA_DIR, "train_labels.npy")
)

val_coords = np.load(
    os.path.join(DATA_DIR, "val_coords.npy")
)

val_labels = np.load(
    os.path.join(DATA_DIR, "val_labels.npy")
)

print("Training coordinates:", train_coords.shape)
print("Validation coordinates:", val_coords.shape)

print("Training positives:", int(train_labels.sum()))
print("Training negatives:", int((train_labels == 0).sum()))

print("Validation positives:", int(val_labels.sum()))
print("Validation negatives:", int((val_labels == 0).sum()))


# ============================================================
# REDUCE VALIDATION
# ============================================================

if len(val_coords) > MAX_VAL_SAMPLES:

    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        len(val_coords),
        size=MAX_VAL_SAMPLES,
        replace=False
    )

    val_coords = val_coords[indices]
    val_labels = val_labels[indices]

print("\nValidation reduced to:", len(val_coords))


# ============================================================
# NORMALIZATION
# ============================================================

print("\n========== LOADING NORMALIZATION ==========\n")

means = np.load(
    os.path.join(STATS_DIR, "feature_means.npy")
).astype(np.float64)

stds = np.load(
    os.path.join(STATS_DIR, "feature_stds.npy")
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
        patch_size=31
    ):

        self.coords = coords
        self.labels = labels

        self.means = means
        self.stds = stds

        self.patch_size = patch_size
        self.radius = patch_size // 2

        self.src = rasterio.open(raster_path)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        y, x = self.coords[idx]

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

        # ----------------------------------------------------
        # NoData handling
        # ----------------------------------------------------

        mask = self.src.read_masks(
            1,
            window=window
        )

        invalid = mask == 0

        for b in range(patch.shape[0]):

            band = patch[b]

            bad = invalid | ~np.isfinite(band)

            band[bad] = self.means[b]

            patch[b] = band

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        patch = (
            patch - self.means[:, None, None]
        ) / self.stds[:, None, None]

        # Protect against extreme values.
        patch = np.clip(
            patch,
            -10.0,
            10.0
        )

        patch = patch.astype(np.float32)

        label = np.float32(self.labels[idx])

        return (
            torch.from_numpy(patch),
            torch.tensor(label)
        )


# ============================================================
# DATASETS
# ============================================================

print("\n========== CREATING DATASETS ==========\n")

train_dataset = GeoDataset(
    train_coords,
    train_labels,
    FEATURES_PATH,
    means,
    stds,
    PATCH_SIZE
)

val_dataset = GeoDataset(
    val_coords,
    val_labels,
    FEATURES_PATH,
    means,
    stds,
    PATCH_SIZE
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# U-NET STYLE MODEL
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


class FaultUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------- Encoder ----------------

        self.enc1 = ConvBlock(19, 32)

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)

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

        # Final feature map

        self.final_conv = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        # Global center prediction

        self.center_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                1,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.25),

            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool3(e3)
        )

        # Decoder

        d3 = self.up3(b)

        # Handle odd spatial dimensions.

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

        logits_map = self.final_conv(d1)

        # Center region instead of exact single pixel.
        #
        # This keeps the model focused on the central
        # geological structure.

        h, w = logits_map.shape[-2:]

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

        return center_logits


# ============================================================
# CREATE MODEL
# ============================================================

print("\n========== CREATING U-NET ==========\n")

model = FaultUNet().to(device)

print(model)


# ============================================================
# LOSS
# ============================================================

# Training set is balanced, so use ordinary BCE initially.
#
# We will later introduce a competition-oriented loss.

criterion = nn.BCEWithLogitsLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

print("\n========== TRAINING ==========\n")


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for patches, targets in progress:

        patches = patches.to(device)

        targets = targets.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(patches)

        loss = criterion(
            logits.squeeze(1),
            targets
        )

        loss.backward()

        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * len(targets)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = (
        running_loss /
        len(train_dataset)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    predictions = []
    true_labels = []

    with torch.no_grad():

        for patches, targets in val_loader:

            patches = patches.to(device)

            targets_device = targets.to(device)

            logits = model(patches)

            loss = criterion(
                logits.squeeze(1),
                targets_device
            )

            val_loss_total += (
                loss.item() *
                len(targets)
            )

            probs = torch.sigmoid(
                logits
            ).squeeze(1)

            predictions.extend(
                probs.cpu().numpy()
            )

            true_labels.extend(
                targets.numpy()
            )

    val_loss = (
        val_loss_total /
        len(val_dataset)
    )

    predictions = np.asarray(
        predictions
    )

    true_labels = np.asarray(
        true_labels
    )

    # Metrics

    try:

        roc_auc = roc_auc_score(
            true_labels,
            predictions
        )

    except ValueError:

        roc_auc = float("nan")

    try:

        pr_auc = average_precision_score(
            true_labels,
            predictions
        )

    except ValueError:

        pr_auc = float("nan")

    scheduler.step(val_loss)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.6f}"
    )

    print(
        f"Val Loss:   {val_loss:.6f}"
    )

    print(
        f"ROC-AUC:    {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:     {pr_auc:.6f}"
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Learning Rate: {current_lr:.6g}"
    )

    # Save best model

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "means": means,

                "stds": stds,

                "patch_size": PATCH_SIZE
            },
            os.path.join(
                MODEL_DIR,
                "fault_unet_spatial_best.pt"
            )
        )

        print("Best model saved.")


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========== FINAL EVALUATION ==========\n")

model.eval()

predictions = []
true_labels = []

with torch.no_grad():

    for patches, targets in val_loader:

        patches = patches.to(device)

        logits = model(patches)

        probs = torch.sigmoid(
            logits
        ).squeeze(1)

        predictions.extend(
            probs.cpu().numpy()
        )

        true_labels.extend(
            targets.numpy()
        )

predictions = np.asarray(
    predictions
)

true_labels = np.asarray(
    true_labels
)

roc_auc = roc_auc_score(
    true_labels,
    predictions
)

pr_auc = average_precision_score(
    true_labels,
    predictions
)

print(
    f"ROC-AUC: {roc_auc:.6f}"
)

print(
    f"PR-AUC:  {pr_auc:.6f}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

final_path = os.path.join(
    MODEL_DIR,
    "fault_unet_spatial.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "means": means,

        "stds": stds,

        "patch_size": PATCH_SIZE
    },
    final_path
)

print("\n========== MODEL SAVED ==========\n")

print(final_path)

print("\n========== DONE ==========\n")