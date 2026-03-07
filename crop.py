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


def get_crop_window(frame, pos, target_w, target_h):
    """Calculate the crop window (x, y, w, h) centered on pos with target aspect ratio."""
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h

    if w / h > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_ratio)

    if pos is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = int(round(pos[0])), int(round(pos[1]))

    half_w = crop_w // 2
    half_h = crop_h // 2
    cx = max(half_w, min(w - half_w, cx))
    cy = max(half_h, min(h - half_h, cy))

    x = cx - half_w
    y = cy - half_h
    return x, y, crop_w, crop_h


def center_crop(frame, target_w, target_h, zoom=1.0):
    """Crop frame to target aspect ratio from center."""
    x, y, w, h = get_crop_window(frame, None, target_w, target_h)
    cropped = frame[y:y + h, x:x + w]
    return _apply_zoom(cropped, zoom)


def subject_crop(frame, pos, target_w, target_h, zoom=1.0):
    """Crop centered on a detected subject. Falls back to center_crop if pos is None."""
    x, y, w, h = get_crop_window(frame, pos, target_w, target_h)
    cropped = frame[y:y + h, x:x + w]
    return _apply_zoom(cropped, zoom)


def get_content_center(frame):
    """Calculate the edge-detection center of mass for the frame with downscaling optimization."""
    h, w = frame.shape[:2]
    # Downscale for much faster processing on CPU
    scale = 0.5
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cx = (M["m10"] / M["m00"]) / scale
        cy = (M["m01"] / M["m00"]) / scale
        return cx, cy
    return w // 2, h // 2


def content_aware_crop(frame, target_w, target_h, zoom=1.0):
    """Crop based on edge-detection center of mass."""
    pos = get_content_center(frame)
    return subject_crop(frame, pos, target_w, target_h, zoom)


class SmoothCropManager:
    """Stateful cropper that smooths movement between frames to avoid jitter."""

    def __init__(self, smoothing=0.12):
        self.smoothing = float(smoothing)
        self.current_pos = None
        self.last_v_path = None
        self.velocity_x = 0.0

    def process(self, frame, target_w, target_h, zoom, v_path, ideal_pos=None):
        """
        Calculate smoothed crop with lead room and apply it.
        """
        h, w = frame.shape[:2]
        if ideal_pos is None:
            ideal_pos = (w / 2, h / 2)

        # Reset if video source changed to avoid drifting between clips
        if v_path != self.last_v_path or self.current_pos is None:
            self.current_pos = np.array(ideal_pos, dtype=np.float32)
            self.last_v_path = v_path
            self.velocity_x = 0.0
        else:
            # --- LEAD ROOM LOGIC ---
            curr_v_x = ideal_pos[0] - self.current_pos[0]
            # Smoothly track velocity
            self.velocity_x = self.velocity_x * 0.9 + curr_v_x * 0.1
            
            # Offset current_pos in direction of velocity to provide lead room
            # Max offset is 15% of frame width
            lead_offset_x = np.clip(self.velocity_x * 5.0, -w * 0.15, w * 0.15)
            
            target = np.array(ideal_pos, dtype=np.float32)
            # Add lead room to the target before smoothing
            target[0] = np.clip(target[0] + lead_offset_x, 0, w)
            
            # Smoothly move current_pos toward target
            self.current_pos += (target - self.current_pos) * self.smoothing

        x, y, cw, ch = get_crop_window(frame, self.current_pos, target_w, target_h)
        cropped = frame[y:y + ch, x:x + cw]
        return _apply_zoom(cropped, zoom)
