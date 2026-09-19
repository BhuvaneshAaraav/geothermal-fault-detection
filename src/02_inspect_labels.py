import rasterio
import numpy as np


FEATURE_FILE = "data/raw/training_features.tif"
LABEL_FILE = "data/raw/Training_fault_labels.tif"


def inspect(path, name):

    with rasterio.open(path) as src:

        print(f"\n========== {name} ==========\n")

        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("Transform:", src.transform)
        print("Data type:", src.dtypes)

        data = src.read(1, masked=True)

        print("\nStatistics:")
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())
        print("Masked pixels:", np.ma.count_masked(data))

        values = data.compressed()

        unique, counts = np.unique(values, return_counts=True)

        print("\nUnique values:")
        for value, count in zip(unique[:20], counts[:20]):
            print(f"  {value}: {count}")


def compare_files():

    with rasterio.open(FEATURE_FILE) as features, \
         rasterio.open(LABEL_FILE) as labels:

        print("\n========== COMPARISON ==========\n")

        print("Same width:",
              features.width == labels.width)

        print("Same height:",
              features.height == labels.height)

        print("Same CRS:",
              features.crs == labels.crs)

        print("Same resolution:",
              features.res == labels.res)

        print("Same bounds:",
              features.bounds == labels.bounds)

        print("Same transform:",
              features.transform == labels.transform)


if __name__ == "__main__":

    inspect(FEATURE_FILE, "FEATURES")

    inspect(LABEL_FILE, "FAULT LABELS")

    compare_files()