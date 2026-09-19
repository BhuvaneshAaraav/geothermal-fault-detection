import rasterio


FILE = "data/raw/training_features.tif"


def inspect_raster(path):

    with rasterio.open(path) as src:

        print("\n========== RASTER INFORMATION ==========\n")

        print("File:", path)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)

        print("\n========== BAND DESCRIPTIONS ==========\n")

        for band in range(1, src.count + 1):

            description = src.descriptions[band - 1]

            print(f"Band {band}: {description}")

        print("\n========== BAND TAGS ==========\n")

        for band in range(1, src.count + 1):

            tags = src.tags(band)

            print(f"\nBand {band}")
            print(tags)


if __name__ == "__main__":
    inspect_raster(FILE)