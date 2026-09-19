import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_RASTER = Path("data/raw/training_features.tif")
OUTPUT_DIR = Path("frontend/public/layers")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BAND NAMES
# ============================================================

FALLBACK_NAMES = [
    "mag_anom",
    "rtp",
    "tmi_hg",
    "geod_2ndinv",
    "iso_grav_anom_slope",
    "tc",
    "geod_shearrate",
    "geod_dilaterate",
    "tmi_vg",
]


def clean_name(name: str | None, index: int) -> str:
    """
    Convert a raster band description into a safe filename.
    """

    if not name:
        if index <= len(FALLBACK_NAMES):
            return FALLBACK_NAMES[index - 1]

        return f"band_{index:02d}"

    name = name.strip().lower()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "(": "",
        ")": "",
        ".": "_",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_band(data: np.ndarray) -> np.ndarray:
    """
    Convert the scientific raster values into an 8-bit image.

    We use percentile clipping so extreme values do not destroy
    the visual contrast of the rest of the layer.
    """

    data = data.astype(np.float32)

    valid = np.isfinite(data)

    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.uint8)

    values = data[valid]

    low = np.percentile(values, 2)
    high = np.percentile(values, 98)

    if high <= low:
        high = low + 1e-6

    normalized = (data - low) / (high - low)

    normalized = np.clip(normalized, 0.0, 1.0)

    normalized = normalized * 255.0

    normalized[~valid] = 0

    return normalized.astype(np.uint8)


# ============================================================
# RGBA IMAGE
# ============================================================

def create_rgba_image(gray: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Create a transparent RGBA image.

    Valid raster pixels become visible.
    NoData pixels become transparent.
    """

    alpha = np.where(
        np.isfinite(original),
        220,
        0,
    ).astype(np.uint8)

    rgba = np.zeros(
        (gray.shape[0], gray.shape[1], 4),
        dtype=np.uint8,
    )

    # Use the normalized value as a neutral grayscale layer.
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = alpha

    return rgba


# ============================================================
# GEOREFERENCING
# ============================================================

def get_web_corners(src):
    """
    Convert the four raster corners from the source CRS
    into WGS84 longitude/latitude.

    Returns MapLibre-compatible corners.
    """

    left = src.bounds.left
    right = src.bounds.right
    top = src.bounds.top
    bottom = src.bounds.bottom

    xs = [
        left,
        right,
        right,
        left,
    ]

    ys = [
        top,
        top,
        bottom,
        bottom,
    ]

    lon, lat = transform(
        src.crs,
        "EPSG:4326",
        xs,
        ys,
    )

    return {
        "topLeft": [
            lon[0],
            lat[0],
        ],
        "topRight": [
            lon[1],
            lat[1],
        ],
        "bottomRight": [
            lon[2],
            lat[2],
        ],
        "bottomLeft": [
            lon[3],
            lat[3],
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GEODAWN GEOPHYSICAL LAYER CONVERTER")
    print("=" * 70)

    if not INPUT_RASTER.exists():
        raise FileNotFoundError(
            f"Input raster not found:\n{INPUT_RASTER}"
        )

    print()
    print("Input:")
    print(INPUT_RASTER)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    with rasterio.open(INPUT_RASTER) as src:

        print()
        print("Raster information")
        print("-" * 70)

        print(f"Bands       : {src.count}")
        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")

        corners = get_web_corners(src)

        print()
        print("Geographic corners:")
        print(json.dumps(corners, indent=2))

        layers = []

        print()
        print("=" * 70)
        print("CONVERTING BANDS")
        print("=" * 70)

        for band_index in range(1, src.count + 1):

            description = src.descriptions[band_index - 1]

            name = clean_name(
                description,
                band_index,
            )

            print()
            print(
                f"[{band_index:02d}/{src.count:02d}] "
                f"{description or name}"
            )

            data = src.read(
                band_index,
                masked=False,
            ).astype(np.float32)

            # Handle raster nodata value.
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            # Also handle infinity.
            data[~np.isfinite(data)] = np.nan

            valid = np.isfinite(data)

            if np.any(valid):

                print(
                    f"    min : {np.nanmin(data):.6g}"
                )

                print(
                    f"    max : {np.nanmax(data):.6g}"
                )

                print(
                    f"    mean: {np.nanmean(data):.6g}"
                )

            else:

                print("    WARNING: no valid pixels")

            normalized = normalize_band(data)

            rgba = create_rgba_image(
                normalized,
                data,
            )

            output_file = (
                OUTPUT_DIR /
                f"{name}.png"
            )

            Image.fromarray(
                rgba,
                mode="RGBA",
            ).save(
                output_file,
                optimize=True,
            )

            print(
                f"    saved: {output_file}"
            )

            layers.append(
                {
                    "index": band_index,
                    "name": name,
                    "description": (
                        description
                        or name
                    ),
                    "file": f"{name}.png",
                }
            )

    # ========================================================
    # WRITE METADATA
    # ========================================================

    metadata = {
        "source": str(INPUT_RASTER),
        "crs": "EPSG:4326",
        "source_crs": str(src.crs),
        "width": src.width,
        "height": src.height,
        "resolution": list(src.res),
        "corners": corners,
        "layers": layers,
    }

    metadata_file = (
        OUTPUT_DIR /
        "layers.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        f"Created {len(layers)} geophysical layers."
    )

    print(
        f"Metadata: {metadata_file}"
    )

    print()
    print("Files:")
    
    for layer in layers:
        print(
            f"  {layer['index']:02d} "
            f"{layer['name']}.png"
        )


if __name__ == "__main__":
    main()