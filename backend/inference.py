import time
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ResUNetA
from config import MODEL_PATH, IMG_SIZE, DEVICE
from logger import logger


# -------------------------------
# 🔧 Transform (USED ONLY ON TILES)
# -------------------------------
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])


# -------------------------------
# 🔧 Sharpen Kernel
# -------------------------------
SHARPEN_KERNEL = np.array([
    [0, -1,  0],
    [-1, 10, -1],
    [0, -1,  0]
], dtype=np.float32)


# -------------------------------
# 🚀 Load model ONCE
# -------------------------------
logger.info(f"Loading model on device: {DEVICE}")

model = ResUNetA(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

logger.info("✅ Model loaded and set to eval mode")


# =========================================================
# ✅ SMALL IMAGE INFERENCE (<=512px)
# =========================================================
def predict_small_image(image_np: np.ndarray) -> np.ndarray:
    logger.info("🧠 Using single-tile inference")

    aug = transform(image=image_np)
    tensor = aug["image"].unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    binary = (pred > 0.5).astype(np.uint8) * 255
    sharp = cv2.filter2D(binary, -1, SHARPEN_KERNEL)

    return sharp

def create_overlay(image_np, mask, color=(0, 255, 0), alpha=0.4):
    """
    Overlay mask on top of original image for better visualization.
    color = (B, G, R)
    alpha = transparency
    """
    overlay = image_np.copy()
    mask_bool = mask > 0

    # Apply the color only on mask pixels
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) +
        np.array(color) * alpha
    ).astype(np.uint8)

    return overlay


# =========================================================
# ✅ LARGE IMAGE INFERENCE (TILING)
# =========================================================
def predict_large_image(image_np: np.ndarray, tile_size: int = 256) -> np.ndarray:
    h, w, _ = image_np.shape
    logger.info(f"🧩 Tiled inference | image={h}x{w} | tile={tile_size}")

    full_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):

            tile = image_np[y:y + tile_size, x:x + tile_size]

            # ✅ Skip incomplete edge tiles (no padding for now)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            aug = transform(image=tile)
            tensor = aug["image"].unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred = model(tensor)[0, 0].cpu().numpy()

            bin_tile = (pred > 0.5).astype(np.uint8) * 255
            full_mask[y:y + tile_size, x:x + tile_size] = bin_tile

    # ✅ Sharpen AFTER stitching
    full_mask = cv2.filter2D(full_mask, -1, SHARPEN_KERNEL)

    return full_mask


# =========================================================
# ✅ MAIN ENTRY POINT
# =========================================================
def predict_from_pil(pil_image: Image.Image):
    start_time = time.time()

    image_np = np.array(pil_image.convert("RGB"))
    h, w, _ = image_np.shape

    logger.info(f"📥 Received image | shape={image_np.shape}")

    # Strategy selection
    if max(h, w) > 512:
        logger.info("🧩 Large image detected → tiled inference")
        mask = predict_large_image(image_np)
    else:
        logger.info("🧠 Small image detected → single inference")
        mask = predict_small_image(image_np)

    # 🔥 FIX: Resize mask to match original image dimension
    if mask.shape != (h, w):
        logger.warning(f"Resizing mask from {mask.shape} to {(h, w)} to match image size")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    elapsed = (time.time() - start_time) * 1000
    logger.info(f"✅ Inference completed in {elapsed:.2f} ms")

    # Create visualization overlay (green semi-transparent)
    overlay = create_overlay(image_np, mask, color=(0, 255, 0), alpha=0.4)

    return mask, overlay

# def predict_from_pil(pil_image: Image.Image) -> np.ndarray:
#     start_time = time.time()

#     image_np = np.array(pil_image.convert("RGB"))
#     h, w, _ = image_np.shape

#     logger.info(f"📥 Received image | shape={image_np.shape}")

#     # ✅ Strategy selection
#     if max(h, w) > 512:
#         logger.info("🧩 Large image detected → tiled inference")
#         mask = predict_large_image(image_np)
#     else:
#         logger.info("🧠 Small image detected → single inference")
#         mask = predict_small_image(image_np)
    
#     elapsed = (time.time() - start_time) * 1000
#     logger.info(f"✅ Inference completed in {elapsed:.2f} ms")

#     # Create visualization overlay (green semi-transparent)
#     overlay = create_overlay(image_np, mask, color=(0, 255, 0), alpha=0.4)

#     return mask, overlay
