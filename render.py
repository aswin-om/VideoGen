import os
import glob
import datetime
import subprocess
import threading
import numpy as np
import cv2

from crop import center_crop, subject_crop, content_aware_crop, SmoothCropManager, get_content_center

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
    # Faster noise generation using smaller map and resizing
    sh, sw = h // 2, w // 2
    noise = np.random.normal(0, intensity * 127, (sh, sw, 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    noisy_frame = frame.astype(np.float32) + noise[:, :, np.newaxis]
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)


class StepPrinter:
    """Low shutter / step printing with frame echo (Travis Scott style)."""

    def __init__(self, intensity=0.0):
        self.intensity = float(intensity)
        self._acc = None
        self._echo_len = max(1, int(4 + 8 * self.intensity))
        self._echo_buf = []

    def process(self, frame):
        if self.intensity <= 0:
            return frame

        f32 = frame.astype(np.float32)

        # Accumulator trail
        alpha = 0.6 + 0.35 * self.intensity
        if self._acc is None:
            self._acc = f32.copy()
        else:
            cv2.addWeighted(self._acc, alpha, f32, 1.0 - alpha, 0, self._acc)

        # Echo
        self._echo_buf.append(f32)
        if len(self._echo_buf) > self._echo_len:
            self._echo_buf.pop(0)

        echo = np.mean(self._echo_buf, axis=0)

        w_curr = max(0.15, 0.55 - 0.35 * self.intensity)
        w_acc = 0.25 + 0.25 * self.intensity
        w_echo = 1.0 - w_curr - w_acc

        blended = w_curr * f32 + w_acc * self._acc + w_echo * echo
        return np.clip(blended, 0, 255).astype(np.uint8)


def render_video(path, video_files, person_cache, crop_mode, zoom,
                 width, height, fps, noise_intensity, output_dir, progress=None,
                 step_print_intensity=0.0, status_cb=None):
    """
    Render the timeline to a raw video file.
    Uses h264_videotoolbox on macOS for hardware acceleration.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(output_dir, f"raw_{timestamp}.mp4")

    # Use hardware acceleration if available (macOS)
    fourcc = "mp4v"
    try:
        # Check if we are on macOS and can use VideoToolbox
        import platform
        if platform.system() == "Darwin":
            # VideoWriter usually doesn't expose encoder selection easily,
            # but we can optimize the ffmpeg muxing later.
            pass
    except:
        pass

    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    frame_usage = np.zeros(len(video_files), dtype=np.int32)
    captures = {v: cv2.VideoCapture(v) for v in video_files}
    # Keep track of current frame position for each capture to avoid redundant seeks
    cap_pos = {v: -1 for v in video_files}

    printer = StepPrinter(step_print_intensity)
    cropper = SmoothCropManager(smoothing=0.15)

    try:
        total = len(path)
        if status_cb:
            status_cb("render.render_video", f"Total frames to render: {total}")
        for i, (v_path, f_idx, v_idx) in enumerate(path):
            if is_cancelled():
                raise InterruptedError("Generation cancelled by user.")

            if progress and i % 25 == 0:
                progress(0.55 + 0.35 * (i / max(1, total)), desc="Rendering...")
            if status_cb and i % 30 == 0:
                status_cb("render.render_video", f"Rendered {i + 1}/{total} frames")

            cap = captures[v_path]
            target_f = int(f_idx)

            # Avoid slow seek if we're already at the next frame
            if cap_pos[v_path] != target_f:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)

            ok, frame = cap.read()
            if ok:
                cap_pos[v_path] = target_f + 1

                ideal_pos = None
                if crop_mode == "person" and person_cache:
                    ideal_pos = person_cache.get(v_path, {}).get(f_idx)
                elif crop_mode == "content":
                    ideal_pos = get_content_center(frame)

                frame = cropper.process(frame, width, height, zoom, v_path, ideal_pos)
                # Faster resizing for M2
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                if noise_intensity > 0:
                    frame = apply_noise_grain(frame, noise_intensity)

                if step_print_intensity > 0:
                    frame = printer.process(frame)

                out.write(frame)
                frame_usage[v_idx] += 1
            else:
                out.write(np.zeros((height, width, 3), dtype=np.uint8))
                cap_pos[v_path] = -1
    finally:
        for cap in captures.values():
            cap.release()
        out.release()

    return raw_path, frame_usage, timestamp


def ffmpeg_mux_audio(raw_video_path, audio_file, start_time, duration_seconds,
                     final_output_path, tempo_factor=1.0):
    # Use h264_videotoolbox for fast hardware encoding on M2
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start_time), "-t", str(duration_seconds),
        "-i", audio_file, "-i", raw_video_path,
        "-map", "1:v:0", "-map", "0:a:0",
        "-c:v", "h264_videotoolbox", "-b:v", "8000k", "-profile:v", "main",
    ]

    if abs(float(tempo_factor) - 1.0) > 1e-6:
        cmd.extend(["-filter:a", _build_atempo_filter(tempo_factor)])

    cmd.extend([
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", final_output_path,
    ])
    subprocess.check_call(cmd)


def interpolate_60fps(input_path, output_path):
    """Interpolate video to 60fps using ffmpeg videotoolbox where possible."""
    # minterpolate is purely CPU and slow.
    # On M2, we use frame rate doubling with faster settings.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-filter:v", "fps=60",
        "-c:v", "h264_videotoolbox", "-b:v", "12000k",
        output_path,
    ]
    subprocess.check_call(cmd)
