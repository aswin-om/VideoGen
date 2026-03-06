import os
import glob
import datetime
import subprocess
import threading
import numpy as np
import cv2

from crop import center_crop, subject_crop, content_aware_crop

_cancel_event = threading.Event()


def cancel_generation():
    _cancel_event.set()


def reset_cancel():
    _cancel_event.clear()


def is_cancelled():
    return _cancel_event.is_set()


def cleanup_old_generations(output_dir):
    """Remove old raw and final video files."""
    for pattern in ["raw_*.mp4", "final_*.mp4"]:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def apply_noise_grain(frame, intensity=0.0):
    if intensity <= 0:
        return frame
    h, w, c = frame.shape
    noise = np.random.normal(0, intensity * 255, (h, w, 1)).astype(np.float32)
    noisy_frame = frame.astype(np.float32) + noise
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)


class StepPrinter:
    """Low shutter / step printing effect (Travis Scott style).

    Blends the current frame with a decaying accumulator of previous frames,
    producing ghostly motion trails.  ``intensity`` (0–1) controls the decay:
    0 = no effect, 1 = maximum trailing.
    """

    def __init__(self, intensity=0.0):
        self.intensity = float(intensity)
        self._acc = None

    def process(self, frame):
        if self.intensity <= 0:
            return frame

        f32 = frame.astype(np.float32)
        alpha = self.intensity * 0.85  # blend weight for the trail

        if self._acc is None:
            self._acc = f32.copy()
        else:
            self._acc = alpha * self._acc + (1.0 - alpha) * f32

        blended = 0.5 * f32 + 0.5 * self._acc
        return np.clip(blended, 0, 255).astype(np.uint8)


def render_video(path, video_files, person_cache, crop_mode, zoom,
                 width, height, fps, noise_intensity, output_dir, progress=None,
                 step_print_intensity=0.0):
    """
    Render the timeline to a raw video file.
    Returns (raw_path, frame_usage_array, timestamp) or raises on cancel.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(output_dir, f"raw_{timestamp}.mp4")
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_usage = np.zeros(len(video_files), dtype=np.int32)
    captures = {v: cv2.VideoCapture(v) for v in video_files}
    printer = StepPrinter(step_print_intensity)

    try:
        total = len(path)
        for i, (v_path, f_idx, v_idx) in enumerate(path):
            if is_cancelled():
                raise InterruptedError("Generation cancelled by user.")

            if progress and i % 20 == 0:
                progress(0.55 + 0.35 * (i / max(1, total)), desc="Rendering...")

            cap = captures[v_path]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ok, frame = cap.read()

            if ok:
                if crop_mode == "person" and person_cache:
                    pos = person_cache.get(v_path, {}).get(f_idx)
                    frame = subject_crop(frame, pos, width, height, zoom)
                elif crop_mode == "content":
                    frame = content_aware_crop(frame, width, height, zoom)
                else:
                    frame = center_crop(frame, width, height, zoom)

                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                if noise_intensity > 0:
                    frame = apply_noise_grain(frame, noise_intensity)

                frame = printer.process(frame)

                out.write(frame)
                frame_usage[v_idx] += 1
            else:
                out.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        for cap in captures.values():
            cap.release()
        out.release()

    return raw_path, frame_usage, timestamp


def ffmpeg_mux_audio(raw_video_path, audio_file, start_time, duration_seconds,
                     final_output_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_time), "-t", str(duration_seconds),
        "-i", audio_file, "-i", raw_video_path,
        "-map", "1:v:0", "-map", "0:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest", final_output_path,
    ]
    subprocess.check_call(cmd)


def interpolate_60fps(input_path, output_path):
    """Interpolate video to 60fps using ffmpeg minterpolate."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-filter:v", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        output_path,
    ]
    subprocess.check_call(cmd)
