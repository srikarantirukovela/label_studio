# utils/georef.py

import rasterio
import cv2
import numpy as np


def write_mask_with_same_georef(
    src_path: str,
    mask: np.ndarray,
    out_path: str,
):
    """
    Write mask in SAME FORMAT as input file.
    - If source has CRS + transform → write georeferenced raster (any driver Rasterio supports).
    - If source has NO CRS → write plain image (no georef).
    """

    with rasterio.open(src_path) as src:
        crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        driver = src.driver

    # Resize mask if needed
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Geo-referenced output
    if crs:
        with rasterio.open(
            out_path,
            "w",
            driver=driver,
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)
    else:
        # No CRS → plain image (no georeferencing)
        cv2.imwrite(out_path, mask)
