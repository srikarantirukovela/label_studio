from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tempfile
import os
from utils.georef import write_mask_with_same_georef
from inference import predict_from_pil
from mask_to_vector import raster_mask_to_shapefile_zip
from logger import logger

import rasterio
import io
import cv2

app = FastAPI(title="Auto Label Backend")

# ✅ CORS (safe for local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LAST_MASK_PATH = None
LAST_MASK_SUFFIX = None
LAST_SHAPE_ZIP = None
LAST_ORIGINAL_NAME = None


@app.post("/preview")
async def preview_image(file: UploadFile = File(...)):
    """Return PNG preview for any image (TIFF included)"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    global LAST_MASK_PATH, LAST_MASK_SUFFIX, LAST_SHAPE_ZIP, LAST_ORIGINAL_NAME

    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1].lower()
    original_name = os.path.splitext(file.filename)[0]   # e.g., "building_23"
    LAST_ORIGINAL_NAME = original_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_tmp:
        src_tmp.write(contents)
        src_path = src_tmp.name

    pil_img = Image.open(io.BytesIO(contents))
    mask = predict_from_pil(pil_img)

    out_mask_path = src_path.replace(suffix, f"_mask{suffix}")

    try:
        # ✅ Preserve georef
        write_mask_with_same_georef(src_path, mask, out_mask_path)

        # ✅ Convert to shapefile
        shp_zip_path = out_mask_path.replace(suffix, "_vector.zip")
        raster_mask_to_shapefile_zip(out_mask_path, shp_zip_path)

        LAST_MASK_PATH = out_mask_path
        LAST_MASK_SUFFIX = suffix
        LAST_SHAPE_ZIP = shp_zip_path

        # ✅ Return preview PNG
        _, png = cv2.imencode(".png", mask)
        return Response(png.tobytes(), media_type="image/png")

    finally:
        os.remove(src_path)


# @app.get("/download-mask")
# def download_mask():
#     if not LAST_OUTPUT_PATH or not os.path.exists(LAST_OUTPUT_PATH):
#         return Response("No output available", status_code=404)

#     ext = LAST_OUTPUT_SUFFIX.lstrip(".")
#     if ext in ("tif", "tiff"):
#         media_type = "image/tiff"
#     elif ext in ("jpg", "jpeg"):
#         media_type = "image/jpeg"
#     elif ext == "png":
#         media_type = "image/png"
#     else:
#         media_type = "application/octet-stream"

#     return FileResponse(
#         LAST_OUTPUT_PATH,
#         media_type=media_type,
#         filename=f"predicted_mask{LAST_OUTPUT_SUFFIX}",
#     )
@app.get("/download-mask")
def download_mask():
    if not LAST_MASK_PATH or not os.path.exists(LAST_MASK_PATH):
        return Response("No mask available", status_code=404)

    return FileResponse(
        LAST_MASK_PATH,
        filename=f"{LAST_ORIGINAL_NAME}_mask{LAST_MASK_SUFFIX}"
    )

@app.get("/download-shapefile")
def download_shapefile():
    if not LAST_SHAPE_ZIP or not os.path.exists(LAST_SHAPE_ZIP):
        return Response("No shapefile available", status_code=404)

    return FileResponse(
        LAST_SHAPE_ZIP,
        filename=f"{LAST_ORIGINAL_NAME}_shapefile.zip"
    )




# ✅ Frontend (mount AFTER routes, and NOT at "/")
app.mount("/ui", StaticFiles(directory="../frontend", html=True), name="ui")

# @app.post("/infer")
# async def infer_image(file: UploadFile = File(...)):
#     logger.info(f"➡️ /infer | {file.filename}")

#     contents = await file.read()
#     suffix = os.path.splitext(file.filename)[1].lower()  # e.g. ".tif", ".png", ".jpg"

#     # Save uploaded file with same extension
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_tmp:
#         src_tmp.write(contents)
#         src_path = src_tmp.name

#     # Run model
#     image = Image.open(io.BytesIO(contents))
#     mask = predict_from_pil(image)  # np.ndarray (H, W)

#     # Output path with same extension
#     out_path = src_path.replace(suffix, f"_mask{suffix}")

#     try:
#         # write result + CRS if present
#         write_mask_with_same_georef(src_path, mask, out_path)

#         # proper media type
#         ext = suffix.lstrip(".")
#         if ext in ("tif", "tiff"):
#             media_type = "image/tiff"
#         elif ext in ("jpg", "jpeg"):
#             media_type = "image/jpeg"
#         elif ext == "png":
#             media_type = "image/png"
#         else:
#             media_type = "application/octet-stream"

#         return FileResponse(
#             out_path,
#             media_type=media_type,
#             filename=f"predicted_mask{suffix}",
#         )

#     finally:
#         # clean up source tmp
#         if os.path.exists(src_path):
#             os.remove(src_path)


