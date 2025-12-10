import os
import tempfile
import zipfile
import rasterio
import rasterio.features
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from logger import logger


def raster_mask_to_shapefile_zip(
    mask_raster_path: str,
    out_zip_path: str,
    class_value: int = 1
) -> str:
    """
    Convert a binary raster mask to a shapefile ZIP
    """
    logger.info("🔁 Converting mask raster → shapefile")

    with rasterio.open(mask_raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    if crs is None:
        raise RuntimeError("Raster CRS missing — cannot vectorize")

    binary = (mask > 0).astype(np.uint8)

    shapes = rasterio.features.shapes(
        binary,
        mask=binary.astype(bool),
        transform=transform
    )

    geometries = []
    classes = []

    for geom, value in shapes:
        if value == 1:
            geometries.append(shape(geom))
            classes.append(class_value)

    if not geometries:
        raise ValueError("No polygons extracted from mask")

    gdf = gpd.GeoDataFrame(
        {"class_id": classes},
        geometry=geometries,
        crs=crs
    )

    tmp_dir = tempfile.mkdtemp(prefix="mask_vector_")
    shp_path = os.path.join(tmp_dir, "mask_polygon.shp")

    logger.info("💾 Writing shapefile")
    gdf.to_file(shp_path)

    logger.info("📦 Zipping shapefile")
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in os.listdir(tmp_dir):
            zipf.write(os.path.join(tmp_dir, f), f)

    logger.info(f"✅ Shapefile ZIP created: {out_zip_path}")
    return out_zip_path
