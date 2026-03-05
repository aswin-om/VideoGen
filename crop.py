import numpy as np
import cv2


def _apply_zoom(cropped, zoom):
    """Apply zoom by cropping a smaller center region."""
    zoom_safe = max(1.0, float(zoom))
    if zoom_safe <= 1.0:
        return cropped
    h, w = cropped.shape[:2]
    new_w = max(1, int(w / zoom_safe))
    new_h = max(1, int(h / zoom_safe))
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    return cropped[y:y + new_h, x:x + new_w]


def center_crop(frame, target_w, target_h, zoom=1.0):
    """Crop frame to target aspect ratio from center."""
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h

    if w / h > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_ratio)

    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    cropped = frame[y:y + crop_h, x:x + crop_w]
    return _apply_zoom(cropped, zoom)


def subject_crop(frame, pos, target_w, target_h, zoom=1.0):
    """Crop centered on a detected subject. Falls back to center_crop if pos is None."""
    if pos is None:
        return center_crop(frame, target_w, target_h, zoom)

    h, w = frame.shape[:2]
    target_ratio = target_w / target_h

    if w / h > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_ratio)

    cx, cy = int(round(pos[0])), int(round(pos[1]))
    half_w = crop_w // 2
    half_h = crop_h // 2
    cx = max(half_w, min(w - half_w, cx))
    cy = max(half_h, min(h - half_h, cy))

    x = cx - half_w
    y = cy - half_h
    cropped = frame[y:y + crop_h, x:x + crop_w]
    return _apply_zoom(cropped, zoom)


def content_aware_crop(frame, target_w, target_h, zoom=1.0):
    """Crop based on edge-detection center of mass."""
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    return subject_crop(frame, (cx, cy), target_w, target_h, zoom)
