"""Image decoding and validation helpers."""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException, UploadFile


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB


async def read_image_bytes(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded image file.

    Args:
        file: Multipart upload from the client.

    Returns:
        Raw image bytes.

    Raises:
        HTTPException: When content type, size, or emptiness is invalid.
    """
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{content_type}'. Use JPEG, PNG, WEBP, or BMP.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds the 12 MB size limit.")
    return data


def decode_image(data: bytes) -> np.ndarray:
    """
    Decode image bytes into a BGR OpenCV ndarray.

    Args:
        data: Raw image bytes.

    Returns:
        BGR image array with shape (H, W, 3).

    Raises:
        HTTPException: When decoding fails.
    """
    import cv2

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    return image
