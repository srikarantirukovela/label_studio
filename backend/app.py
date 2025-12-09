from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tempfile
import os
from utils.georef import write_mask_with_same_georef

import rasterio
import io
import cv2

from inference import predict_from_pil
from logger import logger

app = FastAPI(title="Auto Label Backend")

# ✅ CORS (safe for local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/preview")
async def preview_image(file: UploadFile = File(...)):
    """Return PNG preview for any image (TIFF included)"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
LAST_OUTPUT_PATH = None
LAST_OUTPUT_SUFFIX = None

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    global LAST_OUTPUT_PATH, LAST_OUTPUT_SUFFIX

    logger.info(f"➡️ /infer | {file.filename}")

    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1].lower()

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_tmp:
        src_tmp.write(contents)
        src_path = src_tmp.name

    image = Image.open(io.BytesIO(contents))
    mask = predict_from_pil(image)

    # ✅ keep original-format output for DOWNLOAD
    out_path = src_path.replace(suffix, f"_mask{suffix}")

    try:
        write_mask_with_same_georef(src_path, mask, out_path)

        # ✅ remember for download
        LAST_OUTPUT_PATH = out_path
        LAST_OUTPUT_SUFFIX = suffix

        # ✅ RETURN PNG PREVIEW ONLY
        _, png = cv2.imencode(".png", mask)
        return Response(
            content=png.tobytes(),
            media_type="image/png"
        )

    finally:
        if os.path.exists(src_path):
            os.remove(src_path)

@app.get("/download-mask")
def download_mask():
    if not LAST_OUTPUT_PATH or not os.path.exists(LAST_OUTPUT_PATH):
        return Response("No output available", status_code=404)

    ext = LAST_OUTPUT_SUFFIX.lstrip(".")
    if ext in ("tif", "tiff"):
        media_type = "image/tiff"
    elif ext in ("jpg", "jpeg"):
        media_type = "image/jpeg"
    elif ext == "png":
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        LAST_OUTPUT_PATH,
        media_type=media_type,
        filename=f"predicted_mask{LAST_OUTPUT_SUFFIX}",
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


