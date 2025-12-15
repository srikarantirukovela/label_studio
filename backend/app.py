# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import tempfile
import os
import io
import shutil
import time
import threading
import uuid
import json
import numpy as np
import cv2
from PIL import Image
from logger import logger

# Reuse your utilities if present
try:
    from utils.georef import write_mask_with_same_georef
except Exception as e:
    logger.warning("utils.georef.write_mask_with_same_georef not found; ensure it's available.")
    def write_mask_with_same_georef(src_path, mask_array, out_mask_path):
        from PIL import Image
        im = Image.fromarray(mask_array.astype(np.uint8))
        im.save(out_mask_path)
        logger.warning("Fallback write_mask_with_same_georef used — no georef copied.")

try:
    from mask_to_vector import raster_mask_to_shapefile_zip
except Exception:
    logger.warning("mask_to_vector.raster_mask_to_shapefile_zip not found; shapefile generation may fail.")
    def raster_mask_to_shapefile_zip(mask_raster_path, out_zip_path):
        raise RuntimeError("raster_mask_to_shapefile_zip helper missing. Please provide it.")


app = FastAPI(title="Auto Label Backend (Session + Editable Masks)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_TMP_DIR = os.path.abspath(os.environ.get("BACKEND_TMP_DIR", "./backend_tmp"))
os.makedirs(BASE_TMP_DIR, exist_ok=True)
SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", str(60 * 60 * 6)))  # 6 hours
_sessions_lock = threading.Lock()

def load_original_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except:
        logger.exception("Failed loading original for overlay")
        return None

def make_session_dir(session_id: str) -> str:
    path = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "history"), exist_ok=True)
    return path

def session_path(session_id: str, filename: str) -> str:
    return os.path.join(make_session_dir(session_id), filename)

def now_ts() -> float:
    return time.time()

def cleanup_loop():
    logger.info(f"Session cleaner started, base dir={BASE_TMP_DIR}, ttl={SESSION_TTL}s")
    while True:
        try:
            now = now_ts()
            for name in os.listdir(BASE_TMP_DIR):
                path = os.path.join(BASE_TMP_DIR, name)
                if not os.path.isdir(path):
                    continue
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    continue
                if now - mtime > SESSION_TTL:
                    try:
                        shutil.rmtree(path)
                        logger.info(f"Cleaned old session dir: {path}")
                    except Exception as e:
                        logger.exception(f"Failed to remove {path}: {e}")
        except Exception as e:
            logger.exception("Error in cleanup loop: %s", e)
        time.sleep(60 * 5)

_cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
_cleanup_thread.start()

def _save_blob_to_path(blob_bytes: bytes, out_path: str):
    with open(out_path, "wb") as f:
        f.write(blob_bytes)
    return out_path

def _encode_png_preview(mask_array: np.ndarray) -> bytes:
    _, png = cv2.imencode(".png", mask_array)
    return png.tobytes()

def _atomic_replace_with_fsync(temp_path: str, dest_path: str):
    """
    Atomically replace dest_path with temp_path ensuring data flushed to disk.
    """
    # Ensure dest dir exists
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    # flush temp file to disk
    fd = None
    try:
        fd = os.open(temp_path, os.O_RDWR)
        os.fsync(fd)
    except Exception:
        logger.exception("fsync on temp file failed; proceeding with replace")
    finally:
        if fd:
            os.close(fd)
    # Replace
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.replace(temp_path, dest_path)
    except Exception as e:
        logger.exception("Atomic replace failed: %s", e)
        raise

def _list_history_versions(session_dir: str) -> List[str]:
    hist_dir = os.path.join(session_dir, "history")
    if not os.path.exists(hist_dir):
        return []

    files = os.listdir(hist_dir)

    # Extract numeric index from filenames like v0001_mask.png
    def extract_index(fname: str):
        try:
            # fname format expected: vXXXX_mask.png
            return int(fname.split("_")[0][1:])
        except:
            return 0  # fallback: push unknown files to front

    # Sort numerically, not lexicographically
    files = sorted(files, key=extract_index)
    return files


@app.post("/preview")
async def preview_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    contents = await file.read()
    suffix = os.path.splitext(file.filename)[1].lower() or ".png"
    original_name = os.path.splitext(file.filename)[0]
    session_id = uuid.uuid4().hex
    session_dir = make_session_dir(session_id)

    src_path = os.path.join(session_dir, f"original{suffix}")
    _save_blob_to_path(contents, src_path)

    # Read georeference if available
    src_meta = {}
    try:
        import rasterio
        with rasterio.open(src_path) as ds:
            src_meta["crs"] = ds.crs
            src_meta["transform"] = ds.transform
            src_meta["width"] = ds.width
            src_meta["height"] = ds.height
        with open(os.path.join(session_dir, "meta.json"), "w") as mf:
            json.dump({k: str(v) for k, v in src_meta.items()}, mf)
    except Exception:
        logger.info("No georef in uploaded image")

    # Run inference
    try:
        from inference import predict_from_pil
        pil_img = Image.open(io.BytesIO(contents))
        mask, overlay = predict_from_pil(pil_img)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Encode overlay
    try:
        _, overlay_buf = cv2.imencode(".png", overlay)
        overlay_png_bytes = overlay_buf.tobytes()
    except Exception:
        logger.exception("Overlay encoding failed, using mask as preview")
        overlay_png_bytes = cv2.imencode(".png", mask)[1].tobytes()

    # Encode mask
    mask_png_bytes = cv2.imencode(".png", mask)[1].tobytes()

    # Save georeferenced mask
    mask_suffix = suffix
    mask_path = os.path.join(session_dir, f"mask{mask_suffix}")

    try:
        tmp_mask_path = mask_path + ".tmp"
        write_mask_with_same_georef(src_path, mask, tmp_mask_path)
        _atomic_replace_with_fsync(tmp_mask_path, mask_path)
    except Exception:
        logger.exception("Georef mask write failed, using PNG fallback")
        tmp_png = os.path.join(session_dir, "mask.png.tmp")
        with open(tmp_png, "wb") as f:
            f.write(mask_png_bytes)
            f.flush()
            os.fsync(f.fileno())
        dest_png = os.path.join(session_dir, "mask.png")
        _atomic_replace_with_fsync(tmp_png, dest_png)
        mask_path = dest_png

    # Vectorize mask
    shp_zip_path = os.path.join(session_dir, "vector.zip")
    try:
        raster_mask_to_shapefile_zip(mask_path, shp_zip_path)
    except Exception:
        logger.exception("Vectorization failed")
        shp_zip_path = None

    # Save version history v0001
    hist_dir = os.path.join(session_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)
    v_path = os.path.join(hist_dir, "v0001_mask.png")
    with open(v_path, "wb") as vf:
        vf.write(mask_png_bytes)

    # Write state.json
    state = {
        "session_id": session_id,
        "created_at": now_ts(),
        "original_name": original_name,
        "original_path": src_path,
        "mask_path": mask_path,
        "mask_suffix": mask_suffix,
        "shp_zip": shp_zip_path if shp_zip_path and os.path.exists(shp_zip_path) else None,
        "last_version": "v0001_mask.png"
    }
    with open(os.path.join(session_dir, "state.json"), "w") as sf:
        json.dump(state, sf)

    # Encode final preview
    import base64
    overlay_b64 = base64.b64encode(overlay_png_bytes).decode("ascii")
    mask_b64 = base64.b64encode(mask_png_bytes).decode("ascii")

    return JSONResponse({
        "session_id": session_id,
        "overlay_png_b64": overlay_b64,
        "mask_png_b64": mask_b64,   # <<<<<< IMPORTANT FIX
        "mask_width": mask.shape[1],
        "mask_height": mask.shape[0],
    })



@app.post("/save-edited-mask")
async def save_edited_mask(session_id: str = Query(...), file: UploadFile = File(...)):
    session_dir = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found or expired")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded mask")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, bin_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as mf:
            try:
                meta = json.load(mf)
                width = int(meta.get("width")) if meta.get("width") else None
                height = int(meta.get("height")) if meta.get("height") else None
            except Exception:
                width = None; height = None
    else:
        width = None; height = None

    if width and height:
        if bin_mask.shape[1] != width or bin_mask.shape[0] != height:
            logger.info("Resizing edited mask to original dims %dx%d (was %dx%d)", width, height, bin_mask.shape[1], bin_mask.shape[0])
            bin_mask = cv2.resize(bin_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    state_path = os.path.join(session_dir, "state.json")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=500, detail="Session state missing")
    with open(state_path, "r") as sf:
        state = json.load(sf)

    original_path = state.get("original_path")
    mask_suffix = state.get("mask_suffix") or os.path.splitext(original_path)[1]
    final_mask_path = os.path.join(session_dir, f"mask{mask_suffix}")

    # write to temp then replace atomically
    try:
        tmp_out = final_mask_path + ".tmp"
        try:
            write_mask_with_same_georef(original_path, bin_mask, tmp_out)
            # ensure fsync + replace
            _atomic_replace_with_fsync(tmp_out, final_mask_path)
            logger.info("Edited mask written and replaced: %s", final_mask_path)
        except Exception:
            logger.exception("write_mask_with_same_georef failed on save-edited-mask; writing PNG fallback")
            _, png = cv2.imencode(".png", bin_mask)
            tmp_png = os.path.join(session_dir, "mask.png.tmp")
            with open(tmp_png, "wb") as f:
                f.write(png.tobytes())
                f.flush(); os.fsync(f.fileno())
            dest_png = os.path.join(session_dir, "mask.png")
            _atomic_replace_with_fsync(tmp_png, dest_png)
            final_mask_path = dest_png
    except Exception as e:
        logger.exception("Failed to persist edited mask: %s", e)
        raise HTTPException(status_code=500, detail="Failed to persist edited mask")

    shp_zip_path = os.path.join(session_dir, f"vector.zip")
    try:
        raster_mask_to_shapefile_zip(final_mask_path, shp_zip_path)
        logger.info("Regenerated shapefile at %s", shp_zip_path)
    except Exception:
        logger.exception("raster_mask_to_shapefile_zip failed (may be missing or geo missing)")
        if os.path.exists(shp_zip_path):
            os.remove(shp_zip_path)
        shp_zip_path = None

    hist_dir = os.path.join(session_dir, "history")
    existing_versions = _list_history_versions(session_dir)
    next_idx = len(existing_versions) + 1
    vname = f"v{next_idx:04d}_mask.png"
    vpath = os.path.join(hist_dir, vname)
    _, png = cv2.imencode(".png", bin_mask)
    with open(vpath, "wb") as vf:
        vf.write(png.tobytes())

    # Update state AFTER successful replace
    state.update({
        "mask_path": final_mask_path,
        "shp_zip": shp_zip_path,
        "last_version": vname,
        "updated_at": now_ts()
    })
    with open(state_path, "w") as sf:
        json.dump(state, sf)
    logger.info("State updated for session %s: mask=%s shp=%s version=%s", session_id, final_mask_path, shp_zip_path, vname)

    # preview_png = _encode_png_preview(bin_mask)
    # return Response(content=preview_png, media_type="image/png")
    try:
        original_np = load_original_image(original_path)
        # create overlay only if original available
        if original_np is not None:
            from inference import create_overlay
            overlay = create_overlay(original_np, bin_mask)
            _, overlay_buf = cv2.imencode(".png", overlay)
            overlay_bytes = overlay_buf.tobytes()
        else:
            # fallback: create simple RGB preview from binary
            _, overlay_buf = cv2.imencode(".png", bin_mask)
            overlay_bytes = overlay_buf.tobytes()
    except Exception:
        logger.exception("Failed to build overlay for response; returning mask-only preview")
        _, overlay_buf = cv2.imencode(".png", bin_mask)
        overlay_bytes = overlay_buf.tobytes()

    # Binary mask PNG bytes
    _, mask_png_buf = cv2.imencode(".png", bin_mask)
    mask_png_bytes = mask_png_buf.tobytes()

    import base64
    resp = {
        "mask_png_b64": base64.b64encode(mask_png_bytes).decode("ascii"),
        "overlay_png_b64": base64.b64encode(overlay_bytes).decode("ascii"),
        "mask_width": bin_mask.shape[1],
        "mask_height": bin_mask.shape[0],
        "last_version": vname
    }
    return JSONResponse(resp)




@app.get("/download-mask")
def download_mask(session_id: str = Query(...)):
    session_dir = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    if not os.path.exists(session_dir):
        return Response("Session not found", status_code=404)

    state_path = os.path.join(session_dir, "state.json")
    if not os.path.exists(state_path):
        return Response("No mask available", status_code=404)

    with open(state_path, "r") as sf:
        state = json.load(sf)

    mask_path = state.get("mask_path")
    if not mask_path or not os.path.exists(mask_path):
        return Response("No mask available", status_code=404)

    filename = f"{state.get('original_name', 'output')}_mask{state.get('mask_suffix','')}"
    logger.info("Serving mask download for session %s -> %s", session_id, mask_path)
    headers = {"Cache-Control": "no-store"}
    return FileResponse(mask_path, filename=filename, headers=headers)


@app.get("/download-shapefile")
def download_shapefile(session_id: str = Query(...)):
    session_dir = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    state_path = os.path.join(session_dir, "state.json")
    if not os.path.exists(state_path):
        return Response("Session not found", status_code=404)
    with open(state_path, "r") as sf:
        state = json.load(sf)
    shp_zip = state.get("shp_zip")
    if not shp_zip or not os.path.exists(shp_zip):
        return Response("No shapefile available (vectorization may have failed)", status_code=404)
    filename = f"{state.get('original_name','output')}_shapefile.zip"
    logger.info("Serving shapefile download for session %s -> %s", session_id, shp_zip)
    headers = {"Cache-Control": "no-store"}
    return FileResponse(shp_zip, filename=filename, headers=headers)


@app.get("/history")
def list_versions(session_id: str = Query(...)):
    session_dir = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    versions = _list_history_versions(session_dir)
    return {"versions": versions}

@app.post("/undo")
def undo(session_id: str = Query(...)):
    session_dir = os.path.join(BASE_TMP_DIR, f"session_{session_id}")
    
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")

    hist_dir = os.path.join(session_dir, "history")
    versions = _list_history_versions(session_dir)

    # Need at least 2 versions to undo (current + previous)
    if len(versions) < 2:
        raise HTTPException(status_code=400, detail="No previous version to revert to")

    latest = versions[-1]        # v000X
    prev = versions[-2]          # v000(X-1)
    latest_path = os.path.join(hist_dir, latest)
    prev_path = os.path.join(hist_dir, prev)

    # Load previous version image
    try:
        with open(prev_path, "rb") as pf:
            prev_bytes = pf.read()
        nparr = np.frombuffer(prev_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load previous version image")

    if img is None:
        raise HTTPException(status_code=500, detail="Invalid mask image in history")

    # Convert to binary mask
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, bin_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Read state.json
    state_path = os.path.join(session_dir, "state.json")
    with open(state_path, "r") as sf:
        state = json.load(sf)

    original_path = state.get("original_path")
    mask_suffix = state.get("mask_suffix") or os.path.splitext(original_path)[1]
    out_mask_path = os.path.join(session_dir, f"mask{mask_suffix}")

    # STEP 1 — Write reverted mask safely
    try:
        tmp_out = out_mask_path + ".tmp"
        write_mask_with_same_georef(original_path, bin_mask, tmp_out)
        _atomic_replace_with_fsync(tmp_out, out_mask_path)
        logger.info(f"[UNDO] Successfully reverted mask to: {prev}")
    except Exception:
        logger.exception("[UNDO] georeferenced write failed; falling back to PNG")
        _, png = cv2.imencode(".png", bin_mask)
        tmp_png = os.path.join(session_dir, "mask.png.tmp")
        with open(tmp_png, "wb") as f:
            f.write(png.tobytes())
            f.flush(); os.fsync(f.fileno())
        final_png = os.path.join(session_dir, "mask.png")
        _atomic_replace_with_fsync(tmp_png, final_png)
        out_mask_path = final_png

    # STEP 2 — Regenerate shapefile
    shp_zip_path = os.path.join(session_dir, f"vector.zip")
    try:
        raster_mask_to_shapefile_zip(out_mask_path, shp_zip_path)
        logger.info("[UNDO] Regenerated shapefile after undo")
    except Exception:
        logger.exception("[UNDO] vectorization failed")
        shp_zip_path = None

    # STEP 3 — NOW delete the latest version
    try:
        os.remove(latest_path)
        logger.info(f"[UNDO] Removed version file: {latest}")
    except Exception:
        logger.exception("[UNDO] Failed to remove latest history file")

    # STEP 4 — Update state.json
    state.update({
        "mask_path": out_mask_path,
        "shp_zip": shp_zip_path,
        "last_version": prev,
        "updated_at": now_ts()
    })

    with open(state_path, "w") as sf:
        json.dump(state, sf)

    try:
        original_np = load_original_image(original_path)
        if original_np is not None:
            from inference import create_overlay
            overlay = create_overlay(original_np, bin_mask)
            _, overlay_buf = cv2.imencode(".png", overlay)
            overlay_bytes = overlay_buf.tobytes()
        else:
            _, overlay_buf = cv2.imencode(".png", bin_mask)
            overlay_bytes = overlay_buf.tobytes()
    except Exception:
        logger.exception("Failed to build overlay for undo response; returning mask-only")
        _, overlay_buf = cv2.imencode(".png", bin_mask)
        overlay_bytes = overlay_buf.tobytes()

    _, mask_png_buf = cv2.imencode(".png", bin_mask)
    mask_png_bytes = mask_png_buf.tobytes()

    import base64
    resp = {
        "mask_png_b64": base64.b64encode(mask_png_bytes).decode("ascii"),
        "overlay_png_b64": base64.b64encode(overlay_bytes).decode("ascii"),
        "mask_width": bin_mask.shape[1],
        "mask_height": bin_mask.shape[0],
        "last_version": prev
    }
    return JSONResponse(resp)





frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")
else:
    logger.info("Frontend directory not found at %s — not mounting /ui", frontend_dir)
