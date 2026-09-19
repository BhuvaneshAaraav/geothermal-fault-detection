import os

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = "data/raw/training_features.tif"

DATA_DIR = "data/processed/cnn"

MODEL_DIR = "models"


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 31

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-3

RANDOM_STATE = 42

# First experiment:
# evaluate only 40,000 validation samples.
MAX_VALIDATION_SAMPLES = 40000


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        print("Using Apple MPS GPU")

        return torch.device("mps")

    elif torch.cuda.is_available():

        print("Using CUDA GPU")

        return torch.device("cuda")

    else:

        print("Using CPU")

        return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class FaultPatchDataset(Dataset):

    def __init__(
        self,
        coordinates,
        labels,
        feature_file,
        means,
        stds,
        patch_size=31
    ):

        self.coordinates = coordinates

        self.labels = labels

        self.feature_file = feature_file

        self.means = means.astype(
            np.float64
        )

        self.stds = stds.astype(
            np.float64
        )

        self.patch_size = patch_size

        self.half = patch_size // 2

        self.src = None


    def _open_raster(self):

        if self.src is None:

            self.src = rasterio.open(
                self.feature_file
            )


    def __len__(self):

        return len(
            self.coordinates
        )


    def __getitem__(self, index):

        self._open_raster()

        row, col = self.coordinates[index]


        # ====================================================
        # READ PATCH
        # ====================================================

        window = rasterio.windows.Window(
            col - self.half,
            row - self.half,
            self.patch_size,
            self.patch_size
        )

        # Read the actual data.
        #
        # Shape:
        #
        # (19, 31, 31)

        patch = self.src.read(
            window=window
        )


        # ====================================================
        # READ RASTER MASK
        # ====================================================
        #
        # This is important.
        #
        # Some areas outside the valid geological region
        # can contain huge numerical values.
        #
        # We must use the GeoTIFF mask to identify them.

        mask = self.src.read_masks(
            1,
            window=window
        )


        # ====================================================
        # CONVERT TO FLOAT64
        # ====================================================
        #
        # Do calculations in float64.
        #
        # Only convert to float32 after normalization.

        patch = patch.astype(
            np.float64
        )


        # ====================================================
        # REPLACE MASKED PIXELS
        # ====================================================
        #
        # mask == 0 means invalid/no-data.
        #
        # Replace invalid pixels with the TRAINING MEAN
        # of that feature.
        #
        # This is preferable to inserting an arbitrary zero
        # before normalization.

        invalid = (
            mask == 0
        )

        if np.any(invalid):

            for band in range(
                patch.shape[0]
            ):

                patch[band][invalid] = (
                    self.means[band]
                )


        # ====================================================
        # HANDLE NaN / INF
        # ====================================================

        for band in range(
            patch.shape[0]
        ):

            bad = ~np.isfinite(
                patch[band]
            )

            if np.any(bad):

                patch[band][bad] = (
                    self.means[band]
                )


        # ====================================================
        # GLOBAL NORMALIZATION
        # ====================================================
        #
        # Same mean/std for every patch.
        #
        # Statistics were calculated from training data.

        patch = (
            patch
            - self.means[:, None, None]
        ) / self.stds[:, None, None]


        # ====================================================
        # CLIP EXTREME VALUES
        # ====================================================
        #
        # Normally standardized values are moderate.
        #
        # Clipping protects the neural network from extreme
        # numerical outliers.

        patch = np.clip(
            patch,
            -10.0,
            10.0
        )


        # ====================================================
        # FINAL SAFETY CHECK
        # ====================================================

        patch = np.nan_to_num(
            patch,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )


        # ====================================================
        # FLOAT32
        # ====================================================

        patch = patch.astype(
            np.float32
        )


        # ====================================================
        # PYTORCH TENSORS
        # ====================================================

        x = torch.from_numpy(
            patch
        )

        y = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return x, y


# ============================================================
# CNN MODEL
# ============================================================

class FaultCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                19,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),


            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),


            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2
            ),


            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),


            # ------------------------------------------------
            # Global pooling
            # ------------------------------------------------

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x.squeeze(1)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    total_samples = 0


    for batch_x, batch_y in loader:

        batch_x = batch_x.to(
            device
        )

        batch_y = batch_y.to(
            device
        )


        # Clear gradients

        optimizer.zero_grad()


        # Forward

        logits = model(
            batch_x
        )


        # Loss

        loss = criterion(
            logits,
            batch_y
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Track loss

        batch_size = (
            batch_y.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )


    return (
        total_loss
        / total_samples
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    all_probabilities = []

    all_labels = []


    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                device
            )


            logits = model(
                batch_x
            )


            probabilities = torch.sigmoid(
                logits
            )


            probabilities = (
                probabilities
                .cpu()
                .numpy()
            )


            all_probabilities.append(
                probabilities
            )

            all_labels.append(
                batch_y.numpy()
            )


    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    )


    # ========================================================
    # METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    pr_auc = average_precision_score(
        labels,
        probabilities
    )


    predictions = (
        probabilities >= 0.5
    ).astype(
        np.uint8
    )


    print(
        "\n========== CLASSIFICATION REPORT ==========\n"
    )

    print(
        classification_report(
            labels,
            predictions,
            digits=4
        )
    )


    print(
        "\n========== CONFUSION MATRIX ==========\n"
    )

    print(
        confusion_matrix(
            labels,
            predictions
        )
    )


    print(
        "\nROC-AUC:",
        roc_auc
    )

    print(
        "PR-AUC:",
        pr_auc
    )


    return roc_auc, pr_auc


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(
        RANDOM_STATE
    )

    np.random.seed(
        RANDOM_STATE
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()


    # --------------------------------------------------------
    # Load coordinates
    # --------------------------------------------------------

    print(
        "\n========== LOADING COORDINATES ==========\n"
    )


    train_coords = np.load(
        f"{DATA_DIR}/train_coords.npy"
    )

    train_labels = np.load(
        f"{DATA_DIR}/train_labels.npy"
    )

    val_coords = np.load(
        f"{DATA_DIR}/val_coords.npy"
    )

    val_labels = np.load(
        f"{DATA_DIR}/val_labels.npy"
    )


    print(
        "Training coordinates:",
        train_coords.shape
    )

    print(
        "Validation coordinates:",
        val_coords.shape
    )


    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    print(
        "\n========== LOADING NORMALIZATION ==========\n"
    )


    means = np.load(
        f"{DATA_DIR}/feature_means.npy"
    )

    stds = np.load(
        f"{DATA_DIR}/feature_stds.npy"
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
    # Validation subset
    # --------------------------------------------------------
    #
    # Keep the geographic validation region.
    #
    # We only reduce the number of samples to make the first
    # CNN experiment practical on the 8GB M1 Mac.

    if len(val_coords) > MAX_VALIDATION_SAMPLES:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        selected = rng.choice(
            len(val_coords),
            size=MAX_VALIDATION_SAMPLES,
            replace=False
        )

        val_coords = val_coords[
            selected
        ]

        val_labels = val_labels[
            selected
        ]

        print(
            "\nValidation reduced to:",
            len(val_coords)
        )


    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print(
        "\n========== CREATING DATASETS ==========\n"
    )


    train_dataset = FaultPatchDataset(
        coordinates=train_coords,
        labels=train_labels,
        feature_file=FEATURE_FILE,
        means=means,
        stds=stds,
        patch_size=PATCH_SIZE
    )


    val_dataset = FaultPatchDataset(
        coordinates=val_coords,
        labels=val_labels,
        feature_file=FEATURE_FILE,
        means=means,
        stds=stds,
        patch_size=PATCH_SIZE
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n========== CREATING CNN ==========\n"
    )


    model = FaultCNN()

    model = model.to(
        device
    )


    print(model)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\n========== TRAINING ==========\n"
    )


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )


        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- Loss: {loss:.6f}"
        )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\n========== FINAL EVALUATION ==========\n"
    )


    evaluate(
        model=model,
        loader=val_loader,
        device=device
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    model_path = (
        f"{MODEL_DIR}/"
        "fault_cnn_spatial.pt"
    )


    torch.save(
        model.state_dict(),
        model_path
    )


    print(
        "\n========== MODEL SAVED ==========\n"
    )


    print(
        model_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()