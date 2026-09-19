import rasterio
import matplotlib.pyplot as plt
import numpy as np


FILE = "data/raw/training_features.tif"


def visualize_bands(path):

    with rasterio.open(path) as src:

        for band_number in range(1, src.count + 1):

            band = src.read(band_number, masked=True)

            plt.figure(figsize=(10, 7))

            plt.imshow(band)

            plt.title(f"Feature Band {band_number}")

            plt.colorbar(label="Value")

            plt.xlabel("Column")
            plt.ylabel("Row")

            plt.tight_layout()

            plt.savefig(
                f"outputs/band_{band_number}.png",
                dpi=150
            )

            plt.close()

            print(f"Saved band {band_number}")


if __name__ == "__main__":
    visualize_bands(FILE)